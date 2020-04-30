PY_FILE='/home/cobr/Documents/jcmt-variability-python/NEW_full_cal_test.py'
for region in {IC348,NGC1333,NGC2024,NGC2071,OMC23,OPH_CORE,SERPENS_MAIN,SERPENS_SOUTH}
do
	echo $region
	echo =======
	echo 
	python3 $PY_FILE $region
done
