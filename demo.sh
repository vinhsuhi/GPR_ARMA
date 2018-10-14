STARTDATE='2015-02-02'
MODEL='GPR-ARMA'
MODE='train'


python main.py \
--input_path data/vn_index.txt \
--output_path results \
--start_date ${STARTDATE} \
--model ${MODEL} \
--num_dates 377 \
--num_tests 81 \
--mode ${MODE}


#########################################
# statistic on all results
#########################################
STARTDATE='2015-02-02'
MODEL='all'
MODE='statistic'

python main.py \
--input_path data/vn_index.txt \
--output_path results \
--start_date ${STARTDATE} \
--model ${MODEL} \
--num_dates 377 \
--num_tests 81 \
--mode ${MODE} \
--model ${MODEL}


#########################################
# statistic on arma results
#########################################
STARTDATE='2015-02-02'
MODEL='ARMA'
MODE='statistic'

python main.py \
--input_path data/vn_index.txt \
--output_path results \
--start_date ${STARTDATE} \
--model ${MODEL} \
--num_dates 377 \
--num_tests 81 \
--mode ${MODE} \
--model ${MODEL}


#########################################
# statistic on gpr results
#########################################
STARTDATE='2015-02-02'
MODEL='GPR'
MODE='statistic'

python main.py \
--input_path data/vn_index.txt \
--output_path results \
--start_date ${STARTDATE} \
--model ${MODEL} \
--num_dates 377 \
--num_tests 81 \
--mode ${MODE} \
--model ${MODEL}


#########################################
# statistic on gpr-arma results
#########################################
STARTDATE='2015-02-02'
MODEL='GPR-ARMA'
MODE='train'

python main.py \
--input_path data/vn_index.txt \
--output_path results \
--start_date ${STARTDATE} \
--model ${MODEL} \
--num_dates 60 \
--num_tests 2 \
--mode ${MODE} \
--model ${MODEL}