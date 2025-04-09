import csv
# import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import json
from model import get_model
from dataset import get_test_dataloader
from utils import transform_val


def test(path):
    """Start predict testing ans"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    testdir = "data/test"
    test_dataloader = get_test_dataloader(testdir,
                                          transform=transform_val,
                                          batch_size=1,
                                          shuffle=False)

    file = open("pred.json", mode='w')
    cs = open("pred.csv", mode='w', newline="")
    writer = csv.writer(cs)
    writer.writerow(["image_id","pred_label"])
    ans = []
    for (image, id) in tqdm(test_dataloader):
        image = image.to(device)
        #print(id)
        with torch.no_grad():
            output = model(image)
        output = output[0]
        tmp = []
        for i in range(len(output["boxes"])):
            if(output['scores'][i].item()<0.5):
                continue
            x_min, y_min, x_max, y_max = output['boxes'][i].tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # 轉換為 [x, y, width, height]
            ans.append({
                "image_id": id.item(),
                "bbox": bbox,
                "score": output['scores'][i].item(),
                "category_id": output['labels'][i].item()
            })
            tmp.append({
                "image_id": id.item(),
                "x_min": x_min,
                "category_id": output['labels'][i].item()
            })
        if len(tmp) == 0:
            writer.writerow([id.item(),-1])
        else:
            tmp.sort(key=lambda x:x["x_min"])
            ans_str = 0
            for d in tmp:
                ans_str = ans_str * 10 + int(d["category_id"]) - 1
            writer.writerow([id.item(),ans_str])
            
        # pred_boxes = output["boxes"].cpu().numpy()  # 取得邊界框座標
        # pred_labels = output["labels"].cpu().numpy()  # 取得類別標籤
        # pred_scores = output["scores"].cpu().numpy()  # 取得信心分數

        # if i > 20:
        #     break
        # color = (0, 255, 0)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # image = cv2.imread("data/test/"+str(id.item())+".png")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉換顏色順序 (BGR → RGB)

        # for i in range(len(pred_boxes)):
        #     if pred_scores[i] < 0.5:
        #         continue
        #     x1, y1, x2, y2 = map(int, pred_boxes[i])  # 取得座標
        #     label = f"Class {pred_labels[i]}: {pred_scores[i]:.2f}"  # 標籤文字
        
        #     # 畫框
        #     cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        #     # 放上標籤文字
        #     cv2.putText(image, label, (x1, y1 - 10), font, 0.5, color, 2)

        # # 顯示結果
        # plt.figure(figsize=(8, 8))
        # plt.imshow(image)
        # plt.axis("off")
        # plt.show()
    json.dump(ans,file,indent=4, ensure_ascii=False)
    file.close()
    cs.close()
    print("finish test")


if __name__ == "__main__":
    test("model/exp6/exp6_5_final.pth")

