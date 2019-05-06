pip install pyentrp
# !pip install nolds
pip install git+https://github.com/manu-mannattil/nolitsa.git
pip install saxpy
pip install requests_html
pip install fix_yahoo_finance --upgrade --no-cache-dir

def data_creator(ticker):
  import fix_yahoo_finance as yf 
  from saxpy.znorm import znorm
  hist = yf.download(tickers = ticker, period = 'max')
  hist = hist["Close"]
  pc = [(hist[i + 1] - hist[i])/hist[i] for i in range(len(hist) -1)]
  pc2 = znorm(pc)
  pc3 = [np.floor(c) for c in pc2]
  X = ent.util_pattern_space(pc2, lag = 1, dim = 21)
  X.shape
  trainY = X[:,-1]
  trainX = X[:,:-1]
  trainY = np.where(trainY <= -4, True, False)
  drops = np.where(trainY == True)
  return trainX, trainY, drops
  
  tick_list = ["VTI","VOO","VEA","VWO","VTV","VUG",
             "VO","VB","VEU","VIG","VHT","VFH","VPL",
             "VPU","VSS","VGK","VOT","VSS","VAS","VGT",
             "EFA","EWA","EWH","EWG","EWU","EWQ","EWL","EWP",
             "EWD","EWN","EWI","ERUS","UAE","EIS","INDA"]
             
x_stack = []
y_stack = []
for tick in tick_list:
  x, y, bob = data_creator(tick)
  x_stack.append(x)
  y_stack.append(y)

X = np.vstack(x_stack)
Y = np.hstack(y_stack)
print(X.shape, Y.shape)

from imblearn.over_sampling import SMOTE
mask = np.all(np.isnan(X), axis=1) 
X = X[~mask]
Y = Y[~mask]
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X, Y)
print(X_res.shape, Y_res.shape)
trainX, testX, trainY, testY = train_test_split(X_res,Y_res, shuffle = True)
print(collections.Counter(trainY), collections.Counter(testY))


param_test ={
             'num_leaves': sp_randint(30, 200), 
             'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.5, scale=0.5),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
             'bagging_fraction': sp_uniform(loc=0.5,scale=0.5),
             'feature_fraction':sp_uniform(loc=0.5, scale = 0.5)}



fit_params={"early_stopping_rounds": 30, 
            
            "eval_metric" : ['binary_error'], 
            "eval_set" : [(testX, testY)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}

n_HP_points_to_test = 5



#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, random_state = 333, silent=False, 
                         metric = None, n_jobs=4, class_weight='balanced',
                         # np.random.randint(50), 0: 1},
                         n_estimators = 4000, objective = 'binary', is_unbalanced = True)
gs = RandomizedSearchCV(
     estimator = clf, param_distributions=param_test, 
     n_iter = n_HP_points_to_test,
     scoring = ['balanced_accuracy','f1_weighted'],
     cv = 3,
     refit = 'balanced_accuracy',
     random_state = 320,
     verbose = True)



gs.fit(trainX, trainY, **fit_params)

print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

bst = gs.best_estimator_
#print(bst.get_params)

#####calibrating this classifier##############

from sklearn.calibration import CalibratedClassifierCV
calib = CalibratedClassifierCV(base_estimator = bst, cv = 3)
calib.fit(trainX,trainY)

print(roc_auc_score(testY, calib.predict(testX), average = 'weighted'))
print(classification_report(testY, calib.predict(testX), target_names = ["Zero", "One"]))
cm = confusion_matrix(testY, np.round(calib.predict(testX)), labels=[0,1])
print(cm)

def reporter(clf, ticker, recall):
  x, y, b = data_creator(ticker)
  preds = clf.predict_proba(x[b])
  t = np.quantile(preds[:,-1], (1-recall))
  pclass= []
  all_preds = clf.predict_proba(x)
  for i in range(len(x)):
    if all_preds[i,-1] >= t:
      pclass.append(True)
    else:
      pclass.append(False)
  print(roc_auc_score(y,pclass))
  print(classification_report(y,pclass))
  print(confusion_matrix(y, pclass))
      
tick = "IOO" #try some general ETFs from other companies or funds 
recall = 0.8 #you can try 0.9 or some other "minimum" recall requirement for your model. 
reporter(clf = calib, ticker = tick, recall = recall)
reporter(clf = bst, ticker = tick, recall = recall)





