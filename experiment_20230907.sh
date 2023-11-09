python main.py -md Test -s 2  -alg EDF -e 1000000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_m10_l08alp15_EDF_ep1000000 -jbs l08bt10al15m10_ep1000000
###### 学習率が1.0e-3
## perf_threshが0.5
# nbpが100
python main.py -md Train -s 2 -alg AC -dr -lr 1.0e-3 -ep 50000 -m 1.0 -str -nbp 100 -th 0.5 -pj AC_lr03_drl05_nbp100_ep50000 -bpl bp_long_m10_l08alp15_EDF_ep100000 -jbs l08bt10al15m10_ep1000000
python main.py -md Test -s 2 -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -ml AC_lr03_drl05_nbp100_ep50000 -pj AC_lr03_drl05_nbp100_e100000
# nbpが10
python main.py -md Train -s 2 -alg AC -dr -lr 1.0e-3 -ep 50000 -m 1.0 -str -nbp 10 -th 0.5 -pj AC_lr03_drl05_nbp10_ep50000 -bpl bp_long_m10_l08alp15_EDF_ep100000 -jbs l08bt10al15m10_ep1000000
python main.py -md Test -s 2 -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -ml AC_lr03_drl05_nbp10_ep50000 -pj AC_lr03_drl05_nbp10_e100000
## perf_threshが0.3
# nbpが100
python main.py -md Train -s 2 -alg AC -dr -lr 1.0e-3 -ep 50000 -m 1.0 -str -nbp 100 -th 0.3 -pj AC_lr03_drl03_nbp100_nbp100_ep50000 -bpl bp_long_m10_l08alp15_EDF_ep100000 -jbs l08bt10al15m10_ep1000000
python main.py -md Test -s 2 -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -ml AC_lr03_drl03_nbp100_ep50000 -pj AC_lr03_drl03_nbp100_e100000
# nbpが10
python main.py -md Train -s 2 -alg AC -dr -lr 1.0e-3 -ep 50000 -m 1.0 -str -nbp 10 -th 0.3 -pj AC_lr03_drl03_nbp10_ep50000 -bpl bp_long_m10_l08alp15_EDF_ep100000 -jbs l08bt10al15m10_ep1000000
python main.py -md Test -s 2 -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -ml AC_lr03_drl03_nbp10_ep50000 -pj AC_lr03_drl03_nbp10_e100000
###### 学習率が1.0e-5
## perf_threshが0.5
# nbpが100
python main.py -md Train -s 2 -alg AC -dr -lr 1.0e-5 -ep 50000 -m 1.0 -str -nbp 100 -th 0.5 -pj AC_lr05_drl05_nbp100_ep50000 -bpl bp_long_m10_l08alp15_EDF_ep100000 -jbs l08bt10al15m10_ep1000000
python main.py -md Test -s 2 -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -ml AC_lr05_drl05_nbp100_ep50000 -pj AC_lr05_drl05_nbp100_e100000
# nbpが10
python main.py -md Train -s 2 -alg AC -dr -lr 1.0e-5 -ep 50000 -m 1.0 -str -nbp 10 -th 0.5 -pj AC_lr05_drl05_nbp10_ep50000 -bpl bp_long_m10_l08alp15_EDF_ep100000 -jbs l08bt10al15m10_ep1000000
python main.py -md Test -s 2 -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -ml AC_lr05_drl05_nbp10_ep50000 -pj AC_lr05_drl05_nbp10_e100000
## perf_threshが0.3
# nbpが100
python main.py -md Train -s 2 -alg AC -dr -lr 1.0e-5 -ep 50000 -m 1.0 -str -nbp 100 -th 0.3 -pj AC_lr05_drl03_nbp100_ep50000 -bpl bp_long_m10_l08alp15_EDF_ep100000 -jbs l08bt10al15m10_ep1000000
python main.py -md Test -s 2 -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -ml AC_lr05_drl03_nbp100_ep50000 -pj AC_lr05_drl03_nbp100_e100000
# nbpが10
python main.py -md Train -s 2 -alg AC -dr -lr 1.0e-5 -ep 50000 -m 1.0 -str -nbp 10 -th 0.3 -pj AC_lr05_drl03_nbp10_ep50000 -bpl bp_long_m10_l08alp15_EDF_ep100000 -jbs l08bt10al15m10_ep1000000
python main.py -md Test -s 2 -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -ml AC_lr05_drl03_nbp10_ep50000 -pj AC_lr05_drl03_nbp10_e100000
#
#
python main.py -md Test -s 2 -alg EDF -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj EDF_e100000
python main.py -md Test -s 2 -alg FCFS -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj FCFS_e100000

# #
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_m10_l08alp15_EDF_ep100000
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 1.0 -str -l 0.08 -al 3.0 -bt 10.0 -pj bp_long_m10_l08alp30_EDF_ep100000
# #
# python main.py -md Test -s 2  -alg FCFS -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_m10_l08alp15_FCFS_ep100000
# python main.py -md Test -s 2  -alg FCFS -e 100000 -m 1.0 -str -l 0.08 -al 3.0 -bt 10.0 -pj bp_long_m10_l08alp30_FCFS_ep100000
# #
# python main.py -md Test -s 2  -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_m10_l08alp15_AC_lr03_self_alp15_ep50000_ep100000 -ml AC_lr03_self_alp15_ep50000
# python main.py -md Test -s 2  -alg AC -e 100000 -m 1.0 -str -l 0.08 -al 3.0 -bt 10.0 -pj bp_long_m10_l08alp30_AC_lr03_self_alp30_ep50000_ep100000 -ml AC_lr03_self_alp30_ep50000
# #
# #
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 1.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_m10_l08alp15_EDF_ep100000
# #
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 0.1 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_m01_l08alp15_EDF_ep100000
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 8.0 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_m80_l08alp15_EDF_ep100000
# #
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 1.0 -str -l 0.04 -al 1.5 -bt 10.0 -pj bp_long_m10_l04alp15_EDF_ep100000
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 1.0 -str -l 0.01 -al 1.5 -bt 10.0 -pj bp_long_m10_l01alp15_EDF_ep100000
# #
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 1.0 -str -l 0.08 -al 3.0 -bt 10.0 -pj bp_long_m10_l08alp30_EDF_ep100000
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 1.0 -str -l 0.08 -al 4.5 -bt 10.0 -pj bp_long_m10_l08alp45_EDF_ep100000
# #
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 0.7 -str -l 0.08 -al 1.5 -bt 10.0 -pj bp_long_m07_l08alp15_EDF_ep10000
# python main.py -md Test -s 2  -alg EDF -e 100000 -m 0.7 -str -l 0.01 -al 1.5 -bt 10.0 -pj bp_long_m07_l01alp15_EDF_ep10000
# #

# #
# # python main.py -md Train -alg AC -lr 1.0e-3 -ep 50000 -e 1000 -m 1.0 -str -al 1.5 -pj AC_lr03_self_alp15_ep50000
# # python main.py -md Test -s 1  -alg AC -e 1000 -m 1.0 -str -al 1.5 -pj AC_lr03_self_alp15_ep50000
# # python main.py -md Train -alg AC -lr 1.0e-3 -ep 50000 -e 1000 -m 1.0 -str -al 3.0 -pj AC_lr03_self_alp30_ep50000
# # python main.py -md Test -s 1  -alg AC -e 1000 -m 1.0 -str -al 3.0 -pj AC_lr03_self_alp30_ep50000
# # #
# # python main.py -md Train -alg AC -lr 1.0e-5 -ep 50000 -e 1000 -m 1.0 -str -al 1.5 -pj AC_lr05_self_alp15_ep50000
# # python main.py -md Test -s 1  -alg AC -e 1000 -m 1.0 -str -al 1.5 -pj AC_lr05_self_alp15_ep50000
# # python main.py -md Train -alg AC -lr 1.0e-5 -ep 50000 -e 1000 -m 1.0 -str -al 3.0 -pj AC_lr05_self_alp30_ep50000
# # python main.py -md Test -s 1  -alg AC -e 1000 -m 1.0 -str -al 3.0 -pj AC_lr05_self_alp30_ep50000
