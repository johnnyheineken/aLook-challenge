
For potential deployment I decided to create another python script, which can be run from commandline.

    Requirements: Python 3.6, pandas, numpy
    Installation: Just put clf.py, data\_manipulation.py
    and dual_models_all.pickle in one folder
    Returns 
        {prediction.csv} in the same folder, with first column
        indicating row, second ID_user and third is prediction
        optionally also {Xmatrix.csv}, matrix which was used for prediction
        
    Try with 'python clf.py path\to\transactions.txt'



    usage: clf.py [-h] 
                  [--transactions TRANSACTIONS] 
                  [--products PRODUCTS]
                  [--users USERS] 
                  [--refdate REFDATE] 
                  [--saveX SAVEX]

optional arguments:
  -h, --help        show this help message and exit
  --transactions    Address to transactions.txt
  --products        Address to products.txt (optional) 
  --users           Address to users.txt (optional)
  --refdate         Date for which should be prediction made,
                    passed to pd.Timestamp(). 
                    If left, the most current date 
                    in transactions dataset will be taken 
                    (optional)
  --saveX SAVEX     If set to 1, script also saves
                    "Xmatrix.csv", 
                    which was used for churn prediction
                    (optional)