import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from tabulate import tabulate  # Şık tablo için

# Kandilli Rasathanesi'nin 2023 ve 2024 yıllarındaki deprem verilerine ait URL'ler
URLS = {
    "2024": "http://www.koeri.boun.edu.tr/scripts/lst1.asp",  # 2024 yılı
    "2023": "http://www.koeri.boun.edu.tr/scripts/lst3.asp",  # 2023 yılı
}

# Verileri çekme fonksiyonu
def verileri_cek(url):
    try:
        response = requests.get(url)
        response.encoding = "utf-8"
        if response.status_code != 200:
            print(f"Veri çekilemedi! HTTP Durum Kodu: {response.status_code}")
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        pre_data = soup.find("pre").text
        return pre_data
    except Exception as e:
        print(f"Veri çekilirken hata oluştu: {e}")
        return None

# Veriyi işleme fonksiyonu
def verileri_isle(ham_veri):
    satirlar = ham_veri.splitlines()
    veri_satirlari = satirlar[6:]
    islenmis_veri = []
    for satir in veri_satirlari:
        parcalar = satir.split()
        if len(parcalar) < 9:
            continue
        try:
            tarih = parcalar[0]
            saat = parcalar[1]
            enlem = float(parcalar[2])
            boylam = float(parcalar[3])
            derinlik = float(parcalar[4])
            buyukluk = float(parcalar[6])
            yer = " ".join(parcalar[8:])
        except ValueError:
            continue
        try:
            tarih_objesi = datetime.strptime(tarih, "%Y.%m.%d")
            if tarih_objesi.year in [2023, 2024]:
                islenmis_veri.append([tarih, saat, enlem, boylam, derinlik, buyukluk, yer])
        except ValueError:
            continue
    return islenmis_veri

# Büyük DataFrame'ler için Pandas ayarlarını düzenle
pd.set_option("display.max_rows", 100)  # Maksimum gösterilecek satır sayısını artır
pd.set_option("display.max_columns", None)  # Tüm sütunların görünmesini sağla
pd.set_option("display.width", 1000)  # Çıktı genişliğini artır
pd.set_option("display.colheader_justify", "center")  # Başlık hizasını ortala

# Verileri çekme ve işleme
tum_depremler = []
for yil, url in URLS.items():
    ham_veri = verileri_cek(url)
    if ham_veri:
        islenmis_veri = verileri_isle(ham_veri)
        tum_depremler.extend(islenmis_veri)

# DataFrame oluştur
sutunlar = ["Tarih", "Saat", "Enlem", "Boylam", "Derinlik (km)", "Büyüklük", "Yer"]
df_tum = pd.DataFrame(tum_depremler, columns=sutunlar)
df_tum["Derinlik (km)"] = pd.to_numeric(df_tum["Derinlik (km)"], errors='coerce')
df_tum["Büyüklük"] = pd.to_numeric(df_tum["Büyüklük"], errors='coerce')

# Veriyi kaydet ve terminalde görüntüle
if not df_tum.empty:
    dosya_adi = f"tum_depremler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_tum.to_csv(dosya_adi, index=False, encoding="utf-8")

    print("\nİşlenmiş Deprem Verileri (İlk 20 Satır):")
    print(tabulate(df_tum.head(20), headers="keys", tablefmt="fancy_grid"))  # İlk 20 satırı göster

    print("\nİşlenmiş Deprem Verileri (Son 20 Satır):")
    print(tabulate(df_tum.tail(20), headers="keys", tablefmt="fancy_grid"))  # Son 20 satırı göster

    print("\nTüm Veriler (Pandas DataFrame Formatında):")
    print(df_tum)  # Genişletilmiş Pandas ayarları sayesinde tüm veri seti terminale düzgün yansıyacaktır.

    # 1. Tanımlayıcı İstatistikler
    print("\nTanımlayıcı İstatistikler:")
    print(tabulate(df_tum[["Derinlik (km)", "Büyüklük"]].describe(), headers="keys", tablefmt="fancy_grid"))

    # 2. Görselleştirme
    plt.figure(figsize=(12, 6))
    sns.histplot(df_tum["Büyüklük"], bins=20, kde=True, color="blue", label="Büyüklük")
    plt.title("Deprem Büyüklüğü Dağılımı")
    plt.xlabel("Büyüklük")
    plt.ylabel("Frekans")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_tum, x="Büyüklük")
    plt.title("Deprem Büyüklüğü Kutu Grafiği")
    plt.xlabel("Büyüklük")
    plt.show()

    sns.scatterplot(data=df_tum, x="Derinlik (km)", y="Büyüklük", alpha=0.6)
    plt.title("Derinlik ve Büyüklük Dağılımı")
    plt.xlabel("Derinlik (km)")
    plt.ylabel("Büyüklük")
    plt.grid()
    plt.show()

    # 3. Hipotez Testleri: t-testi
    print("\nHipotez Testi:")
    grup1 = df_tum[df_tum["Büyüklük"] >= 4]["Derinlik (km)"]
    grup2 = df_tum[df_tum["Büyüklük"] < 4]["Derinlik (km)"]
    t_stat, p_value = stats.ttest_ind(grup1, grup2, nan_policy='omit')
    print(f"t-İstatistiği: {t_stat:.2f}, p-Değeri: {p_value:.2e}")

    # 4. Korelasyon Analizi
    print("\nKorelasyon Matrisi:")
    korelasyon_matrisi = df_tum[["Derinlik (km)", "Büyüklük"]].corr()
    print(tabulate(korelasyon_matrisi, headers="keys", tablefmt="fancy_grid"))
    sns.heatmap(korelasyon_matrisi, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Korelasyon Matrisi")
    plt.show()

    # 5. Regresyon Analizi
    X = df_tum[["Derinlik (km)"]].dropna().values.reshape(-1, 1)
    y = df_tum["Büyüklük"].dropna().values

    if len(X) > 1 and len(y) > 1:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        print("\nRegresyon Analizi Sonuçları:")
        print(f"Eğim: {model.coef_[0]:.2f}")
        print(f"Kesişim Noktası: {model.intercept_:.2f}")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_tum, x="Derinlik (km)", y="Büyüklük", alpha=0.6, label="Veri")
        plt.plot(X, y_pred, color="red", label="Regresyon Çizgisi")
        plt.title("Regresyon: Deprem Derinliği ve Büyüklüğü")
        plt.xlabel("Derinlik (km)")
        plt.ylabel("Büyüklük")
        plt.legend()
        plt.show()
    else:
        print("Regresyon analizi için yeterli veri yok.")
else:
    print("Veri çekilemedi veya analiz için yeterli veri yok.")