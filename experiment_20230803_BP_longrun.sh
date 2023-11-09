# # lambda = 0.08
python main.py -md Test -s 1  -alg EDF -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_l08alp15_EDF_ep100000
python main.py -md Test -s 1  -alg EDF -e 100000 -m 1.0 -str -l 0.08 -al 3.0 -bt 10.0 -pj bp_long_l08alp30_EDF_ep100000
#
python main.py -md Test -s 1  -alg FCFS -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_l08alp15_FCFS_ep100000
python main.py -md Test -s 1  -alg FCFS -e 100000 -m 1.0 -str -l 0.08 -al 3.0 -bt 10.0 -pj bp_long_l08alp30_FCFS_ep100000
#
python main.py -md Test -s 1  -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_l08alp15_AC_lr03_self_alp15_ep50000_ep100000 -ml AC_lr03_self_alp15_ep50000
python main.py -md Test -s 1  -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 3.0 -bt 10.0 -pj bp_long_l08alp30_AC_lr03_self_alp30_ep50000_ep100000 -ml AC_lr03_self_alp30_ep50000
# # lambda = 0.06
python main.py -md Test -s 1  -alg EDF -e 100000 -m 1.0 -str -l 0.06 -al 1.5 -bt 10.0 -pj bp_long_l06alp15_EDF_ep100000
python main.py -md Test -s 1  -alg EDF -e 100000 -m 1.0 -str -l 0.06 -al 3.0 -bt 10.0 -pj bp_long_l06alp30_EDF_ep100000
#
python main.py -md Test -s 1  -alg FCFS -e 100000 -m 1.0 -str -l 0.06 -al 1.5 -bt 10.0 -pj bp_long_l06alp15_FCFS_ep100000
python main.py -md Test -s 1  -alg FCFS -e 100000 -m 1.0 -str -l 0.06 -al 3.0 -bt 10.0 -pj bp_long_l06alp30_FCFS_ep100000
#
python main.py -md Test -s 1  -alg AC -e 100000 -m 1.0 -str -l 0.06 -al 1.5 -bt 10.0 -pj bp_long_l06alp15_AC_lr03_self_alp15_ep50000_ep100000 -ml AC_lr03_self_alp15_ep50000
python main.py -md Test -s 1  -alg AC -e 100000 -m 1.0 -str -l 0.06 -al 3.0 -bt 10.0 -pj bp_long_l06alp30_AC_lr03_self_alp30_ep50000_ep100000 -ml AC_lr03_self_alp30_ep50000


#
# python main.py -md Train -alg AC -lr 1.0e-3 -ep 50000 -e 1000 -m 1.0 -str -al 1.5 -pj AC_lr03_self_alp15_ep50000
# python main.py -md Test -s 1  -alg AC -e 1000 -m 1.0 -str -al 1.5 -pj AC_lr03_self_alp15_ep50000
# python main.py -md Train -alg AC -lr 1.0e-3 -ep 50000 -e 1000 -m 1.0 -str -al 3.0 -pj AC_lr03_self_alp30_ep50000
# python main.py -md Test -s 1  -alg AC -e 1000 -m 1.0 -str -al 3.0 -pj AC_lr03_self_alp30_ep50000
# #
# python main.py -md Train -alg AC -lr 1.0e-5 -ep 50000 -e 1000 -m 1.0 -str -al 1.5 -pj AC_lr05_self_alp15_ep50000
# python main.py -md Test -s 1  -alg AC -e 1000 -m 1.0 -str -al 1.5 -pj AC_lr05_self_alp15_ep50000
# python main.py -md Train -alg AC -lr 1.0e-5 -ep 50000 -e 1000 -m 1.0 -str -al 3.0 -pj AC_lr05_self_alp30_ep50000
# python main.py -md Test -s 1  -alg AC -e 1000 -m 1.0 -str -al 3.0 -pj AC_lr05_self_alp30_ep50000
