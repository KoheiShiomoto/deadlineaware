#
#python main.py -md Test  -alg EDF -e 15 -tr -n 1 -pj EDF_trace
#python main.py -md Test  -alg FCFS -e 15 -tr -n 1 -pj FCFS_trace
#python main.py -md Train -alg AC -lr 1.0e-3 -ep 100000 -e 15 -tr -n 1 -pj AC_trace_lr03_n1_ep100000
#python main.py -md Train -alg AC -lr 1.0e-5 -ep 100000 -e 15 -tr -n 1 -pj AC_trace_lr05_n1_ep100000
python main.py -md Train -alg AC -lr 1.0e-3 -ep 100000 -e 15 -tr -n 10 -pj AC_trace_lr03_n10_ep100000
python main.py -md Train -alg AC -lr 1.0e-5 -ep 100000 -e 15 -tr -n 10 -pj AC_trace_lr05_n10_ep100000
python main.py -md Train -alg AC -lr 1.0e-3 -ep 100000 -e 15 -tr -n 100 -pj AC_trace_lr03_n100_ep100000
python main.py -md Train -alg AC -lr 1.0e-5 -ep 100000 -e 15 -tr -n 100 -pj AC_trace_lr05_n100_ep100000
