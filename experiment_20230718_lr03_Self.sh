# #
# python main.py -md Test  -alg EDF -e 15 -tr -i data_jobset_sgsk2022K1.txt -pj EDF_traceK1
# python main.py -md Test  -alg EDF -e 30 -tr -i data_jobset_sgsk2022K2.txt -pj EDF_traceK2
# python main.py -md Test  -alg EDF -e 60 -tr -i data_jobset_sgsk2022K4.txt -pj EDF_traceK4
# #
# python main.py -md Test  -alg FCFS -e 15 -tr -i data_jobset_sgsk2022K1.txt -pj FCFS_traceK1
# python main.py -md Test  -alg FCFS -e 30 -tr -i data_jobset_sgsk2022K2.txt -pj FCFS_traceK2
# python main.py -md Test  -alg FCFS -e 60 -tr -i data_jobset_sgsk2022K4.txt -pj FCFS_traceK4
# #
# python main.py -md Train -alg AC -lr 1.0e-3 -ep 50000 -e 15 -tr -i data_jobset_sgsk2022K1.txt -pj AC_traceK1_lr03_ep50000
# python main.py -md Test  -alg AC -e 15 -tr -i data_jobset_sgsk2022K1.txt -pj AC_traceK1_lr03_ep50000
# python main.py -md Train -alg AC -lr 1.0e-3 -ep 50000 -e 30 -tr -i data_jobset_sgsk2022K2.txt -pj AC_traceK2_lr03_ep50000
# python main.py -md Test  -alg AC -e 30 -tr -i data_jobset_sgsk2022K2.txt -pj AC_traceK2_lr03_ep50000
# python main.py -md Train -alg AC -lr 1.0e-3 -ep 50000 -e 60 -tr -i data_jobset_sgsk2022K4.txt -pj AC_traceK4_lr03_ep50000
# python main.py -md Test  -alg AC -e 60 -tr -i data_jobset_sgsk2022K4.txt -pj AC_traceK4_lr03_ep50000
# #
# python main.py -md Train -alg AC -lr 1.0e-5 -ep 50000 -e 15 -tr -i data_jobset_sgsk2022K1.txt -pj AC_traceK1_lr05_ep50000
# python main.py -md Test  -alg AC -e 15 -tr -i data_jobset_sgsk2022K1.txt -pj AC_traceK1_lr05_ep50000
# python main.py -md Train -alg AC -lr 1.0e-5 -ep 50000 -e 30 -tr -i data_jobset_sgsk2022K2.txt -pj AC_traceK2_lr05_ep50000
# python main.py -md Test  -alg AC -e 30 -tr -i data_jobset_sgsk2022K2.txt -pj AC_traceK2_lr05_ep50000
# python main.py -md Train -alg AC -lr 1.0e-5 -ep 50000 -e 60 -tr -i data_jobset_sgsk2022K4.txt -pj AC_traceK4_lr05_ep50000
# python main.py -md Test  -alg AC -e 60 -tr -i data_jobset_sgsk2022K4.txt -pj AC_traceK4_lr05_ep50000
# ##
# ##
# ##
# python main.py -md Test  -alg EDF -e 1000 -str -al 1.5 -pj EDF_self_alp15_ep1000
# python main.py -md Test  -alg EDF -e 1000 -str -al 3.0 -pj EDF_self_alp30_ep1000
# #
# python main.py -md Test  -alg FCFS -e 1000 -str -al 1.5 -pj FCFS_self_alp15_ep1000
# python main.py -md Test  -alg FCFS -e 1000 -str -al 3.0 -pj FCFS_self_alp30_ep1000
# #
python main.py -md Train -alg AC -lr 1.0e-3 -ep 50000 -e 1000 -str -al 1.5 -pj AC_lr03_self_alp15_ep50000
python main.py -md Test  -alg AC -e 1000 -str -al 1.5 -pj AC_lr03_self_alp15_ep50000
python main.py -md Train -alg AC -lr 1.0e-3 -ep 50000 -e 1000 -str -al 3.0 -pj AC_lr03_self_alp30_ep50000
python main.py -md Test  -alg AC -e 1000 -str -al 3.0 -pj AC_lr03_self_alp30_ep50000
# #
# python main.py -md Train -alg AC -lr 1.0e-5 -ep 50000 -e 1000 -str -al 1.5 -pj AC_lr05_self_alp15_ep50000
# python main.py -md Test  -alg AC -e 1000 -str -al 1.5 -pj AC_lr05_self_alp15_ep50000
# python main.py -md Train -alg AC -lr 1.0e-5 -ep 50000 -e 1000 -str -al 3.0 -pj AC_lr05_self_alp30_ep50000
# python main.py -md Test  -alg AC -e 1000 -str -al 3.0 -pj AC_lr05_self_alp30_ep50000
