import numpy as np, time
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

# ---------------- constants (synthetic network, abstract single supply class) ----------------
NB = 6                                   # forward bases (2 hubs x 3 bases)
HUBOF = np.array([0, 0, 0, 1, 1, 1])
MU = np.array([8., 9., 8., 8., 9., 8.])  # mean daily demand
SMIN = 2.0 * MU                          # survival minimum stock
T = 28                                   # episode length (days)
CP, FLEET = 14., 32.                     # per-link and per-hub daily primary lift
L1, L2 = 2, 3                            # lead times: primary / alternate corridor
ALT_NOM, ALT_CONT = 8., 2.               # alternate corridor spot lift: normal / contested (primary closed)
H = 6                                    # planning horizon
E_RES = 4                                # option validity after issue (days)
WU, WV, HOLD, SHIP, FRES = 10., 1., 0.02, 0.02, 0.12
ALPHA = 0.9
P_CLOSE_BASE, P_CLOSE_WX, P_OPEN = 0.04, 0.20, 0.25
SIG = 0.25


# ---------------- environment (common random numbers per seed) ----------------
def make_episode(seed):
    g = np.random.default_rng(seed)
    wx = np.zeros(T + H + 2, bool)
    for _ in range(2):
        s = g.integers(2, T - 6); wx[s:s + 5] = True
    openv = np.ones((NB, T + H + 2), bool)
    U = g.random((NB, T + H + 2))
    for t in range(1, T + H + 2):
        for b in range(NB):
            if openv[b, t - 1]:
                openv[b, t] = not (U[b, t] < (P_CLOSE_WX if wx[t] else P_CLOSE_BASE))
            else:
                openv[b, t] = U[b, t] < P_OPEN
    surge = np.ones((NB, T))
    if g.random() < 0.7:
        s = g.integers(8, 18); bs = g.choice(NB, 2, replace=False)
        for b in bs: surge[b, s:s + 6] = 1.6
    z = g.standard_normal((NB, T))
    dem = MU[:, None] * surge * np.exp(SIG * z - SIG ** 2 / 2)
    blk = np.zeros((NB, T), bool)
    if g.random() < 0.7:
        s = g.integers(8, 20); blk[3:6, s:s + 3] = True
    return dict(wx=wx, open=openv, dem=dem, blk=blk, surge=surge)


# ---------------- forecaster ----------------
class Forecaster:
    def __init__(self):
        self.f = MU.copy(); self.cus = np.zeros(NB); self.pool = []; self.last = [list([m]) for m in MU]
        self.cov_hits = 0; self.cov_n = 0

    def update(self, d, seen, cal):
        for b in range(NB):
            if not seen[b]: continue
            ratio = d[b] / self.f[b]
            if cal:
                self.pool.append(ratio)
                self.cus[b] = max(0., self.cus[b] + (ratio - 1) - 0.15)
            self.last[b].append(d[b]); self.last[b] = self.last[b][-3:]
            self.f[b] = 0.7 * self.f[b] + 0.3 * d[b]
            if cal and self.cus[b] > 1.0:
                self.f[b] = max(self.f[b], np.mean(self.last[b]))
        self.pool = self.pool[-NB * 21:]

    def sample(self, S, g, cal):
        if cal and len(self.pool) >= 12:
            r = np.array(self.pool)
            q = r[g.integers(0, len(r), (S, NB, H))]
            q = q / max(np.mean(r), 1e-6) * 1.0  # keep mean ratio ~1 (bias handled by CUSUM/EWMA)
            return self.f[None, :, None] * q
        z = g.standard_normal((S, NB, H))
        return self.f[None, :, None] * np.exp(SIG * z - SIG ** 2 / 2)


# ---------------- planner: two-stage (CVaR) recourse LP ----------------
def plan(s0, pipe, open_now, f, dem_s, open_s, r_act, lam, use_res, spot_now):
    """s0[B], pipe[B,H+1] known arrivals at start of day k (k=1..H), dem_s[S,B,H], open_s[S,B,H] (k index 0 = today unused).
    Returns xp, xa, r, stats."""
    S = dem_s.shape[0]; K = H - 1
    nvar_g = 3 * NB + 1
    per = NB * K * 2 + NB * H * 3 + 1
    N = nvar_g + S * per
    c = np.zeros(N); lb = np.zeros(N); ub = np.full(N, 1e4)
    def XP(b): return b
    def XA(b): return NB + b
    def R(b): return 2 * NB + b
    Z = 3 * NB
    def base(w): return nvar_g + w * per
    def YP(w, b, k): return base(w) + b * K + (k - 2)            # k=2..H
    def YA(w, b, k): return base(w) + NB * K + b * K + (k - 2)
    def U(w, b, k): return base(w) + 2 * NB * K + b * H + (k - 1)
    def V(w, b, k): return base(w) + 2 * NB * K + NB * H + b * H + (k - 1)
    def SS(w, b, k): return base(w) + 2 * NB * K + 2 * NB * H + b * H + (k - 1)
    def ETA(w): return base(w) + per - 1
    # first-stage bounds & costs
    for b in range(NB):
        ub[XP(b)] = CP * (1.0 if open_now[b] else 0.0)
        ub[XA(b)] = spot_now[b] + r_act[b]
        ub[R(b)] = ALT_NOM if use_res else 0.0
        c[XP(b)] = SHIP; c[XA(b)] = SHIP
        c[R(b)] = FRES * 3
    ub[Z] = 1e5; lb[Z] = 0
    c[Z] = lam
    rows, cols, vals, rhs = [], [], [], []
    eq_r, eq_c, eq_v, eq_b = [], [], [], []
    nr = 0; ne = 0
    def addu(ri, ci, v): rows.append(ri); cols.append(ci); vals.append(v)
    def adde(ri, ci, v): eq_r.append(ri); eq_c.append(ci); eq_v.append(v)
    wgt = 1.0 / S
    for w in range(S):
        for b in range(NB):
            for k in range(2, H + 1):
                spot = ALT_NOM if open_s[w, b, k - 1] else ALT_CONT
                ub[YP(w, b, k)] = CP * (1.0 if open_s[w, b, k - 1] else 0.0)
                ub[YA(w, b, k)] = ALT_NOM if k <= E_RES else spot
                c[YP(w, b, k)] = SHIP * wgt; c[YA(w, b, k)] = SHIP * wgt
                if k <= E_RES and spot < ALT_NOM:
                    addu(nr, YA(w, b, k), 1.0); addu(nr, R(b), -1.0); rhs.append(spot); nr += 1
            for k in range(1, H + 1):
                c[U(w, b, k)] = WU * wgt; c[V(w, b, k)] = WV * wgt; c[SS(w, b, k)] = HOLD * wgt
                # stock balance: s_k - s_{k-1} - arrivals + u = -d
                adde(ne, SS(w, b, k), 1.0)
                if k > 1: adde(ne, SS(w, b, k - 1), -1.0)
                adde(ne, U(w, b, k), -1.0)
                arr_const = pipe[b, k]
                if k == 1 + L1: adde(ne, XP(b), -1.0)
                if k == 1 + L2: adde(ne, XA(b), -1.0)
                for j in range(2, k):
                    if j + L1 == k: adde(ne, YP(w, b, j), -1.0)
                    if j + L2 == k: adde(ne, YA(w, b, j), -1.0)
                eq_b.append(-dem_s[w, b, k - 1] + arr_const + (s0[b] if k == 1 else 0.0)); ne += 1
                # v + s >= SMIN
                addu(nr, V(w, b, k), -1.0); addu(nr, SS(w, b, k), -1.0); rhs.append(-SMIN[b]); nr += 1
        for k in range(2, H + 1):
            for h in (0, 1):
                for b in np.where(HUBOF == h)[0]: addu(nr, YP(w, b, k), 1.0)
                rhs.append(FLEET); nr += 1
        # CVaR: sum(WU u + WV v) - z - eta <= 0
        if lam > 0:
            for b in range(NB):
                for k in range(1, H + 1):
                    addu(nr, U(w, b, k), WU); addu(nr, V(w, b, k), WV)
            addu(nr, Z, -1.0); addu(nr, ETA(w), -1.0); rhs.append(0.0); nr += 1
            c[ETA(w)] = lam / ((1 - ALPHA) * S)
    for h in (0, 1):
        for b in np.where(HUBOF == h)[0]: addu(nr, XP(b), 1.0)
        rhs.append(FLEET); nr += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(nr, N)).tocsr()
    A_eq = coo_matrix((eq_v, (eq_r, eq_c)), shape=(ne, N)).tocsr()
    t0 = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=np.array(rhs), A_eq=A_eq, b_eq=np.array(eq_b),
                  bounds=np.c_[lb, ub], method='highs')
    dt = time.time() - t0
    if res.status != 0:
        return np.zeros(NB), np.zeros(NB), np.zeros(NB), dict(t=dt, vars=N, cons=nr + ne, ok=False)
    x = res.x
    return x[:NB], x[NB:2 * NB], x[2 * NB:3 * NB], dict(t=dt, vars=N, cons=nr + ne, ok=True)


def sample_open(open_now, wxflags, S, g, robust=False):
    """closure scenarios for k=1..H (index k-1). open_s[:,:,0] = current (known)."""
    o = np.ones((S, NB, H), bool); o[:, :, 0] = open_now[None, :]
    for k in range(1, H):
        pc = P_CLOSE_WX if (k < len(wxflags) and k <= 3 and wxflags[k]) else 0.07
        u = g.random((S, NB))
        prev = o[:, :, k - 1]
        o[:, :, k] = np.where(prev, u >= pc, u < P_OPEN)
    return o


# ---------------- policies ----------------
POLICIES = {
    'B1':  dict(kind='ss'),
    'B2':  dict(kind='det'),
    'B4':  dict(kind='robust'),
    'B3':  dict(kind='lp', S=20, lam=0., res=False, cal=True, edge=False),
    'RRL_noOpt': dict(kind='lp', S=20, lam=1., res=False, cal=True, edge=False),
    'RRL_noCal': dict(kind='lp', S=20, lam=1., res=True, cal=False, edge=True),
    'RRL_noEdge': dict(kind='lp', S=20, lam=1., res=True, cal=True, edge=False),
    'RRL': dict(kind='lp', S=20, lam=1., res=True, cal=True, edge=True),
}


def run_episode(seed, pol, tree=None, log=None, S_override=None, no_blackout=False):
    cfg = POLICIES[pol] if isinstance(pol, str) else pol
    ep = make_episode(seed); g = np.random.default_rng(seed + 7919)
    blk = np.zeros_like(ep['blk']) if no_blackout else ep['blk']
    stock = 5.0 * MU.copy(); est = stock.copy()
    pipe_true = np.zeros((NB, T + 10)); pipe_est = np.zeros((NB, T + 10))
    r_act = np.zeros(NB); r_exp = np.full(NB, -1)
    fc = Forecaster(); cal = cfg.get('cal', False)
    m = dict(so_days=0, unmet=0., below=0, stock_sum=0., fee=0., mincover=np.full(NB, 1e9), spells=[], blk_so=0,
             times=[], vars=[], cons=[], cov_hit=0, cov_n=0, blk_days=0)
    in_spell = np.zeros(NB, int)
    for t in range(T):
        visible = ~blk[:, t]
        # arrivals
        stock = stock + pipe_true[:, t]; est = est + pipe_est[:, t]
        # observe state
        open_now = ep['open'][:, t]
        wxf = ep['wx'][t:t + 4]
        pipe_rel = np.zeros((NB, H + 1))
        for k in range(1, H + 1):
            if t + k - 1 < pipe_true.shape[1]: pipe_rel[:, k] = pipe_est[:, t + k - 1] * 0 + pipe_true[:, t + k - 1] * 0
        # known pipeline in relative days (arrivals at start of day t+k-1), strictly future (>t)
        for k in range(2, H + 1):
            pipe_rel[:, k] = pipe_est[:, t + k - 1]
        s_plan = np.where(visible, stock, est)
        open_plan = np.where(visible, open_now, True)
        xp = np.zeros(NB); xa = np.zeros(NB)
        spot_now = np.where(open_now, ALT_NOM, ALT_CONT)
        edge_rows = []
        if cfg['kind'] == 'ss':
            for b in range(NB):
                pos = s_plan[b] + pipe_est[b, t + 1:t + L1 + 1].sum() + pipe_est[b, t + L1 + 1:t + L2 + 1].sum()
                if pos <= 3.5 * fc.f[b]: xp[b] = min(CP, 6.0 * fc.f[b] - pos)
            xp = np.where(open_plan, xp, 0.)
        elif cfg['kind'] in ('det', 'robust'):
            if cfg['kind'] == 'det':
                dem_s = np.repeat(fc.f[None, :, None], 1, 0).repeat(H, 2)
                o_s = np.ones((1, NB, H), bool); o_s[0, :, 0] = open_plan
            else:
                dem_s = (fc.f * 1.25)[None, :, None].repeat(H, 2)
                o_s = np.ones((1, NB, H), bool); o_s[0, :, 0] = open_plan; o_s[0, :, 1:3] = False
            xp, xa, _, st = plan(s_plan, pipe_rel, open_plan, fc.f, dem_s, o_s, r_act * 0, 0., False, spot_now)
            m['times'].append(st['t']); m['vars'].append(st['vars']); m['cons'].append(st['cons'])
        else:
            S = S_override or cfg['S']
            dem_s = fc.sample(S, g, cal)
            o_s = sample_open(open_plan, ep['wx'][t:t + 4], S, g)
            r_cur = np.where(r_exp >= t, r_act, 0.)
            xp, xa, rnew, st = plan(s_plan, pipe_rel, open_plan, fc.f, dem_s, o_s, r_cur, cfg['lam'], cfg['res'], spot_now)
            m['times'].append(st['t']); m['vars'].append(st['vars']); m['cons'].append(st['cons'])
            # reservations only renewed when connected (hub-side authority); frozen in blackout
            for b in range(NB):
                if visible[b] and cfg['res']:
                    r_act[b] = rnew[b]; r_exp[b] = t + E_RES
            if log is not None:
                for b in range(NB):
                    if visible[b]:
                        pipe_b = pipe_est[b, t + 1:t + L2 + 1].sum()
                        lab = 2 if xa[b] > spot_now[b] + 0.25 else (1 if xp[b] > 0.5 and open_now[b] else 0)
                        log.append(([stock[b] / MU[b], float(not open_now[b]), r_act[b] / MU[b], pipe_b / MU[b]], lab))
        # edge override for blacked-out bases
        if cfg.get('edge') and tree is not None:
            for b in range(NB):
                if not visible[b]:
                    pipe_b = pipe_est[b, t + 1:t + L2 + 1].sum()
                    feat = [[stock[b] / MU[b], float(not open_now[b]), (r_act[b] if r_exp[b] >= t else 0.) / MU[b], pipe_b / MU[b]]]
                    a = int(tree.predict(feat)[0])
                    xp[b] = 0.; xa[b] = 0.
                    if a == 2: xa[b] = r_act[b] if r_exp[b] >= t else 0.
                    elif a == 1 and open_now[b]: xp[b] = min(CP, max(0., 5 * MU[b] - stock[b] - pipe_b))
        # execution with physical limits
        r_cur = np.where(r_exp >= t, r_act, 0.)
        xa = np.minimum(xa, spot_now + r_cur)
        xp = np.where(open_now, xp, 0.)
        for h in (0, 1):
            idx = np.where(HUBOF == h)[0]; tot = xp[idx].sum()
            if tot > FLEET: xp[idx] *= FLEET / tot
        xp = np.minimum(xp, CP)
        for b in range(NB):
            if xp[b] > 1e-6:
                pipe_true[b, t + L1] += xp[b]; pipe_est[b, t + L1] += xp[b]
            if xa[b] > 1e-6:
                pipe_true[b, t + L2] += xa[b]; pipe_est[b, t + L2] += xa[b]
        # demand
        d = ep['dem'][:, t]
        unmet = np.maximum(0., d - stock); stock = np.maximum(0., stock - d)
        est = np.maximum(0., est - fc.f)
        est = np.where(visible, stock, est)
        # fee, metrics
        if cfg.get('res'): m['fee'] += FRES * r_cur.sum()
        m['so_days'] += int((unmet > 1e-9).sum()); m['unmet'] += unmet.sum()
        m['below'] += int((stock < SMIN).sum()); m['stock_sum'] += stock.sum()
        m['mincover'] = np.minimum(m['mincover'], stock / MU)
        if blk[:, t].any():
            m['blk_so'] += int(((unmet > 1e-9) & blk[:, t]).sum()); m['blk_days'] += int(blk[:, t].sum())
        for b in range(NB):
            if stock[b] < SMIN[b]: in_spell[b] += 1
            elif in_spell[b] > 0: m['spells'].append(in_spell[b]); in_spell[b] = 0
        # forecast calibration coverage (80% interval from pool) measured before updating
        if cal and len(fc.pool) >= 30 and True:
            lo, hi = np.quantile(fc.pool, 0.1), np.quantile(fc.pool, 0.9)
            for b in range(NB):
                if visible[b]:
                    m['cov_n'] += 1; m['cov_hit'] += int(lo * fc.f[b] <= d[b] <= hi * fc.f[b])
        elif (not cal):
            lo, hi = np.exp(-1.2816 * SIG - SIG ** 2 / 2), np.exp(1.2816 * SIG - SIG ** 2 / 2)
            for b in range(NB):
                if visible[b]:
                    m['cov_n'] += 1; m['cov_hit'] += int(lo * fc.f[b] <= d[b] <= hi * fc.f[b])
        m.setdefault('trace', []).append(stock.copy()); m.setdefault('open_tr', []).append(open_now.copy())
        fc.update(d, visible, cal)
    for b in range(NB):
        if in_spell[b] > 0: m['spells'].append(in_spell[b])
    return m
