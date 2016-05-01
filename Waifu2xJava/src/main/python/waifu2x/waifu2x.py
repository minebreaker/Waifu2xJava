# coding=utf-8

# waifu2x.py by @marcan42 based on https://github.com/nagadomi/waifu2x
# MIT license, see https://github.com/nagadomi/waifu2x/blob/master/LICENSE
import json, sys, numpy as np
from scipy import misc, signal
from PIL import Image
# Model export script: https://mrcn.st/t/export_model.lua (needs a working waifu2x install)

# 入力ファイル、出力ファイル、モデル(json)のパス
infile, outfile, modelpath = sys.argv[1:]

# modelを読み込む
model = json.load(open(modelpath))

# 画像を開いて、YCbCrに変換
im = Image.open(infile).convert("YCbCr")
# 画像をニアレストで2倍に拡大して、NumPy配列に変換
im = misc.fromimage(im.resize((2*im.size[0], 2*im.size[1]), resample=Image.NEAREST)).astype("float32")

# 各ピクセルの最初の要素(=Y)をモデルの長さで埋めて入力平面を生成
# 255で割るのは0-1に収めるため
# また、waifu2xはYCbCrのうちYのみを対象にする
# numpy.pad(array, pad_width, mode=None, **kwargs)
planes = [np.pad(im[:,:,0], len(model), "edge") / 255.0]

# 進行情報を数えるための情報 処理には関係ない
count = sum(step["nInputPlane"] * step["nOutputPlane"] for step in model) # 各ネットの入力x出力
progress = 0 # 進行状況

# stepはネットワークを適応する回数。すなわち7ステップ。
for step in model:

    # ネットの入力数とplanesの数が等しい
    assert step["nInputPlane"] == len(planes)
    # ネットの出力数、各ステップの重みの長さとバイアスの長さが等しい
    assert step["nOutputPlane"] == len(step["weight"]) == len(step["bias"])

    o_planes = [] # 出力

    # zipは複数要素をまとめてイテレートする (個別に、ではなく同時にインクリメントしていく)
    for bias, weights in zip(step["bias"], step["weight"]):

        # 平面の処理対象になるテンポラリな「部分」
        partial = None

        for ip, kernel in zip(planes, weights):
            # 畳みこみ演算
            p = signal.convolve2d(ip, np.float32(kernel), "valid")
            # 計算結果を加算
            if partial is None:
                partial = p
            else:
                partial += p

            # 進行状況を表示
            progress += 1
            sys.stderr.write("\r%.1f%%..." % (100 * progress / float(count)))

        # バイアスを加える
        partial += np.float32(bias)
        # 出力平面にテンポラリな平面を追加
        o_planes.append(partial)

    # << for bias, weights in zip(step["bias"], step["weight"]):

    # 出力平面を次の入力平面として代入
    # ただし、pが0以下の場合0.1倍する
    planes = [np.maximum(p, 0) + 0.1 * np.minimum(p, 0) for p in o_planes]

# << for step in model:

## 最後の出力平面は1つに収束
assert len(planes) == 1

# 0以下または1以上の値はそれぞれ0, 1に丸め、255倍して元に戻す
# numpy.clip(a, a_min, a_max, out=None)
im[:,:,0] = np.clip(planes[0], 0, 1) * 255
# RGBに変換して保存
misc.toimage(im, mode="YCbCr").convert("RGB").save(outfile)

sys.stderr.write("Done\n")
