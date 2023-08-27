#
python main.py -md Test  -alg EDF -e 15 -tr -pj EDF_trace
python main.py -md Test  -alg FCFS -e 15 -tr -pj FCFS_trace
#
python main.py -md Train -alg AC -lr 1.0e-2 -ep 1000 -e 15 -tr -pj AC_trace_lr02_ep1000
python main.py -md Test  -alg AC -lr 1.0e-2 -e 15 -tr -pj AC_trace_lr02_ep1000
python main.py -md Train -alg AC -lr 1.0e-3 -ep 1000 -e 15 -tr -pj AC_trace_lr03_ep1000
python main.py -md Test  -alg AC -lr 1.0e-3 -e 15 -tr -pj AC_trace_lr03_ep1000
python main.py -md Train -alg AC -lr 1.0e-4 -ep 1000 -e 15 -tr -pj AC_trace_lr04_ep1000
python main.py -md Test  -alg AC -lr 1.0e-4  -e 15 -tr -pj AC_trace_lr04_ep1000
python main.py -md Train -alg AC -lr 1.0e-5 -ep 1000 -e 15 -tr -pj AC_trace_lr05_ep1000
python main.py -md Test  -alg AC -lr 1.0e-5 -e 15 -tr -pj AC_trace_lr05_ep1000
#
python main.py -md Train -alg AC -lr 1.0e-2 -ep 5000 -e 15 -tr -pj AC_trace_lr02_ep5000
python main.py -md Test  -alg AC -lr 1.0e-2 -e 15 -tr -pj AC_trace_lr02_ep5000
python main.py -md Train -alg AC -lr 1.0e-3 -ep 5000 -e 15 -tr -pj AC_trace_lr03_ep5000
python main.py -md Test  -alg AC -lr 1.0e-3 -e 15 -tr -pj AC_trace_lr03_ep5000
python main.py -md Train -alg AC -lr 1.0e-4 -ep 5000 -e 15 -tr -pj AC_trace_lr04_ep5000
python main.py -md Test  -alg AC -lr 1.0e-4  -e 15 -tr -pj AC_trace_lr04_ep5000
python main.py -md Train -alg AC -lr 1.0e-5 -ep 5000 -e 15 -tr -pj AC_trace_lr05_ep5000
python main.py -md Test  -alg AC -lr 1.0e-5 -e 15 -tr -pj AC_trace_lr05_ep5000
#
python main.py -md Train -alg AC -lr 1.0e-2 -ep 10000 -e 15 -tr -pj AC_trace_lr02_ep10000
python main.py -md Test  -alg AC -lr 1.0e-2 -e 15 -tr -pj AC_trace_lr02_ep10000
python main.py -md Train -alg AC -lr 1.0e-3 -ep 10000 -e 15 -tr -pj AC_trace_lr03_ep10000
python main.py -md Test  -alg AC -lr 1.0e-3 -e 15 -tr -pj AC_trace_lr03_ep10000
python main.py -md Train -alg AC -lr 1.0e-4 -ep 10000 -e 15 -tr -pj AC_trace_lr04_ep10000
python main.py -md Test  -alg AC -lr 1.0e-4  -e 15 -tr -pj AC_trace_lr04_ep10000
python main.py -md Train -alg AC -lr 1.0e-5 -ep 10000 -e 15 -tr -pj AC_trace_lr05_ep10000
python main.py -md Test  -alg AC -lr 1.0e-5 -e 15 -tr -pj AC_trace_lr05_ep10000
#
python main.py -md Train -alg AC -lr 1.0e-2 -ep 100000 -e 15 -tr -pj AC_trace_lr02_ep100000
python main.py -md Test  -alg AC -lr 1.0e-2 -e 15 -tr -pj AC_trace_lr02_ep100000
python main.py -md Train -alg AC -lr 1.0e-3 -ep 100000 -e 15 -tr -pj AC_trace_lr03_ep100000
python main.py -md Test  -alg AC -lr 1.0e-3 -e 15 -tr -pj AC_trace_lr03_ep100000
python main.py -md Train -alg AC -lr 1.0e-4 -ep 100000 -e 15 -tr -pj AC_trace_lr04_ep100000
python main.py -md Test  -alg AC -lr 1.0e-4  -e 15 -tr -pj AC_trace_lr04_ep100000
python main.py -md Train -alg AC -lr 1.0e-5 -ep 100000 -e 15 -tr -pj AC_trace_lr05_ep100000
python main.py -md Test  -alg AC -lr 1.0e-5 -e 15 -tr -pj AC_trace_lr05_ep100000
