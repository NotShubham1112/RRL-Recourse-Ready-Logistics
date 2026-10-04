from rrl_sim import *
m = run_episode(3, 'B2')    # conventional plan
r = run_episode(3, 'RRL')   # note: pass tree=... to use the edge policy
print(m['so_days'], r['so_days'])