##end toend ml project
## 1.create environment

''' conda create -p venv python==3.8

conda activate "path_of_venv"
 '''

##2.install all package fron requirment.txt

''' pip install -r requirment.txt '''
##3.create src under which all ml project run
''' inside that logging,exception,handling ans utils are create utils having any generic funtionality that wont to create '''

##4.create the EDA and model traning in note book
''' if error occur during running regarding to the ipykernal the run "pip install ipykernal" in cmd to solve it '''
##5. code the logger and exception handling
##6. in src/components do data ingetion (run the data_ingection file check the output)


""coding sequence for project"

#1. setup.py
#after creating setup.py file then install it

#2. logger.py
#3. Exception.py
#4. Utils.py
#5. data_ingection.py
#6. Data_tranformation.py = (all feature engg data cleaning do in this file)
#7.Modeltraniner.py
#8.delete thr __name__=="__main__" part from the data_ingection file and paste into traning pipeline and run the traning pipeline file

