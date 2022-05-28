import json
from utils import *
from config import *
from pysurvival.utils import load_model
import sys

info = sys.argv[1]
feature = sys.argv[2]
res = info.replace("info", "result")

print("INFO:", info)
print("RESULT:", res)
print("FEATURE:", feature)

# Load model
csf = load_model(config["save_path"]+"csf_"+feature+".zip")
rsf = load_model(config["save_path"]+"rsf_"+feature+".zip")

# 入参
with open("./jsonFile/"+info, 'r', encoding='utf8') as fp:
    x_dict = json.load(fp)
print(x_dict)

# 解析
x_list = list(x_dict.values())
x_ndarray = np.array(x_list).reshape(1, -1)

# 预测
prs_csf = csf.predict_survival(x_ndarray).squeeze()
prs_rsf = rsf.predict_survival(x_ndarray).squeeze()
print("CSF:", prs_csf)
print("RSF:", prs_rsf)
prs = (prs_csf + prs_rsf) / 2

prs_list = prs.tolist()
v = 0.5
middle_day = prs_list.index(find_nearest(prs, v))

# 展示的时候把概率保留三位小数
prs_list = [round(i, 3) for i in prs_list]

# 返回生存概率和中位生存时间
result = {"prs": prs_list, "mid_day": middle_day}
jsonStr = json.dumps(result)
print(jsonStr)

# save to file
with open("./jsonFile/"+res, 'w', encoding='utf8') as fp:
    json.dump(result, fp)
