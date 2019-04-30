#install numpy, pandas, fix_yahoo_finance, saxpy, matplotlib

hist = yf.download(tickers = "DJI", period = 'max')
hist = hist["Close"]

rows=[]
for i in range(len(dow_df)):
  row = dow_df[i,:]
  row = znorm(row) 
  rows.append(row)
  
df = np.vstack(rows)

km = TimeSeriesKMeans(n_clusters=6,verbose=True)
y_pred = km.fit_predict(df)

collections.Counter(y_pred)

fig = plt.figure(figsize=(10,8))
for yi in range(6):
    ax = plt.subplot(3, 2, yi + 1)
    for xx in df[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.10)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 50)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("Euclidean $k$-means")
fig.savefig('dowjones_clusters.png')
