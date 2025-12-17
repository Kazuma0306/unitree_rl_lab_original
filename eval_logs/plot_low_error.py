import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("eval_FR.csv")
e = df["final_err"].to_numpy()

# 文字で言いたい指標
p50, p90, p99 = np.quantile(e, [0.50, 0.90, 0.99])
s2 = (e < 0.02).mean()
s3 = (e < 0.03).mean()

print(f"p50={p50*100:.2f} cm, p90={p90*100:.2f} cm, p99={p99*100:.2f} cm")
print(f"success@2cm={s2*100:.1f}%, success@3cm={s3*100:.1f}%")

# CDF (ECDF)
x = np.sort(e)
y = np.arange(1, len(x)+1) / len(x)

plt.figure()
plt.plot(x*100, y)  # cm表示
plt.axvline(2.0, linestyle="--")  # 2cm
plt.axvline(3.0, linestyle="--")  # 3cm
plt.xlabel("final_err [cm]")
plt.ylabel("CDF")
plt.title("Final error CDF")
plt.show()

# Histogram
plt.figure()
plt.hist(e*100, bins=40)
plt.xlabel("final_err [cm]")
plt.ylabel("count")
plt.title("Final error histogram")
plt.show()
