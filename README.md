# 🩻 پروژه تشخیص هوشمند بیماری ذات الریه (Pneumonia Detection System)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

این مخزن شامل سامانه جامع تشخیص خودکار عفونت ریه (پنومونی / ذات‌الریه) از روی تصاویر رادیولوژی قفسه سینه (X-Ray) با استفاده از شبکه عصبی پیچشی اختصاصی (**PneumoniaCNN**) در فریم‌ورک **PyTorch** و مجموعه داده **PneumoniaMNIST** است.

---
---

## 🎯 معرفی پروژه

تشخیص سریع و دقیق بیماری پنومونی از روی تصاویر رادیولوژی قفسه سینه، یکی از چالش‌های کلیدی در حوزه تصویربرداری پزشکی است. این پروژه با هدف پیاده‌سازی یک خط‌لوله (Pipeline) کامل و استاندارد یادگیری عمیق طراحی شده است که شامل موارد زیر است:
- پیش‌پردازش و نرمال‌سازی تصاویر پزشکی به ابعاد ۱۱۲ در ۱۱۲ پیکسل.
- مدیریت عدم توازن داده‌ها با وزن‌دهی تابع زیان (`BCEWithLogitsLoss`).
- آموزش هوشمند همراه با توقف زودهنگام (Early Stopping) و تنظیم خودکار نرخ یادگیری (Scheduler).
- ارائه رابط کاربری تعاملی تحت وب (Streamlit) و نسخه دسکتاپ (GUI).

---

## 📂 ساختار مخزن (Repository Structure)

```text
.
├── artifacts/                  # خروجی‌های مدل، جداول و نمودارها
│   ├── figures/                # نمودارهای آموزشی و ارزیابی گزارش عملکرد
├── data/                       # فایل‌های دانلود شده دیتاست MedMNIST
├── pneumonia_detection.ipynb   # نوت‌بوک اصلی آموزش و ارزیابی خط‌به‌خط
├── requirements.txt            # پیش‌نیازها برای pip
├── environment.yml             # تنظیمات محیط مجازی برای Conda
└── README.md                   #


📊 مجموعه داده (Dataset)این پروژه از مجموعه داده PneumoniaMNIST (بخشی از پکیج MedMNIST) استفاده می‌کند. داده‌ها شامل تصاویر رادیولوژی خاکستری با دو کلاس سالم (Normal) و مبتلا به پنومونی (Pneumonia) هستند.بخش داده (Split)تعداد کل تصاویرسالم (Normal)پنومونی (Pneumonia)درصد پنومونیآموزشی (Train)۴,۷۰۸۱,۲۱۴۳,۴۹۴۷۴.۲۱٪اعتبارسنجی (Val)۵۲۴۱۳۵۳۸۹۷۴.۲۴٪آزمون (Test)۶۲۴۲۳۴۳۹۰۶۲.۵۰٪🧠 معماری مدل (Model Architecture)شبکه طراحی‌شده (PneumoniaCNN) یک شبکه عصبی پیچشی عمیق با ساختار زیر است:PlaintextInput (1, 112, 112)
   │
   ├── Conv2d(32) + BatchNorm + ReLU + MaxPool(2x2) ──► (32, 56, 56)
   ├── Conv2d(64) + BatchNorm + ReLU + MaxPool(2x2) ──► (64, 28, 28)
   ├── Conv2d(128) + BatchNorm + ReLU + MaxPool(2x2) ─► (128, 14, 14)
   │
 Flatten (25,088)
   │
   ├── Dropout(0.3) + Linear(128) + ReLU
   ├── Dropout(0.3) + Linear(1)
   │
Output Logit (Binary Classification)
⚙️ نصب و راه‌اندازی (Setup & Installation)ابتدا پروژه را کلون کرده و وارد پوشه اصلی شوید:Bashgit clone [https://github.com/username/pneumonia-detection.git](https://github.com/username/pneumonia-detection.git)
cd pneumonia-detection
روش ۱: استفاده از Conda (پیشنهادی)Bashconda env create -f environment.yml
conda activate pneumonia_env
روش ۲: استفاده از pipBashpython -m venv venv
# فعال‌سازی در ویندوز:
venv\Scripts\activate
# فعال‌سازی در مک/لینوکس:
source venv/bin/activate

pip install -r requirements.txt
🚀 نحوه اجرا (Usage)۱. اجرای نوت‌بوک آموزش مدلبرای بازسازی کامل مراحل دانلود داده، آموزش مدل و رسم نمودارها:Bashjupyter notebook pneumonia_detection.ipynb
۲. اجرای سامانه وب (Streamlit Web App)برای اجرای محیط تعاملی و آپلود عکس توسط کاربر:Bashstreamlit run app.py
۳. اجرای برنامه گرافیکی دسکتاپ (Desktop App)Bashpython app_gui.py
📈 نتایج و ارزیابی (Evaluation & Results)عملکرد مدل روی داده‌های آزمون (Test Set) با معیارهای استاندارد پزشکی ارزیابی شده است:معیار (Metric)مقدار (Value)دقت کلی (Accuracy)۸۹.۵۸٪حساسیت (Recall / Sensitivity)۹۶.۱۵٪دقت پیش‌بینی (Precision)۸۸.۲۴٪معیار F1-Score۹۲.۰۲٪سطح زیر منحنی (ROC-AUC)۰.۹۴۸۲
---

### فایل فرعی: `artifacts/README.md`

```markdown
# 📌 معرفی پوشه خروجی‌ها (Artifacts Directory)

این پوشه شامل تمام خروجی‌های ساختاریافته، مدل‌های ذخیره‌شده و گزارش‌های بصری حاصل از اجرای پروژه تشخیصی پنومونی است.

## 📝 جزئیات و وظایف فایل‌ها

* 📂 **`figures/`**:
  * `fig01_sample_images.png`: نمایش تصادفی داده‌های آموزشی به همراه برچسب مربوطه.
  * `fig02_performance_summary.png`: نمودارهای افت (Loss Curves)، منحنی ROC و ماتریس اغتشاش (Confusion Matrix).
* 📂 **`tables/`**:
  * `tab01_dataset_summary.csv`: گزارش تعداد و درصد تفکیکی داده‌ها در سه فاز آموزش، اعتبارسنجی و تست.
  * `tab02_test_metrics.csv`: مقادیر عددی دقیق تمامی معیارهای ارزیابی نهایی روی داده‌های تست.
* 🛡️ **`best_model.pth`**:
  * فایل وزن‌های ذخیره‌شده شبکه `PneumoniaCNN` در بهترین اپاک اعتبارسنجی (کمترین مقدار Validation Loss). این فایل برای بارگذاری مدل در برنامه‌های `app.py` و `app_gui.py` استفاده می‌شود.



مستندات اصلی پروژه
