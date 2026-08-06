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



## 📂 ساختار مخزن (Repository Structure)

```text
.
├── figures/                  # خروجی‌های مدل، جداول و نمودارها           
├── pneumonia_detection.ipynb   # نوت‌بوک اصلی آموزش و ارزیابی خط‌به‌خط
├── requirements.txt            # پیش‌نیازها برای pip
├── environment.yml             # تنظیمات محیط مجازی برای Conda
└── README.md                   #
```

📊 مجموعه داده (Dataset)این پروژه از مجموعه داده PneumoniaMNIST (بخشی از پکیج MedMNIST) استفاده می کند.


معماری مدل
```text
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

```


