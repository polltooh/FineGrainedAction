import os

dunk_res = 'test_res_nba_dunk.txt'
jumpshot_res = 'test_res_nba_jumpshot.txt'
layup_res = 'test_res_nba_layup.txt'



with open(dunk_res, 'r') as f:
    s = f.read()
    s_l = s.split('\n') 
    for key, val in s_l:
        
    print(s_l[0])
     
