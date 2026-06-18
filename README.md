Hibridni sistem za semantičku pretragu akademskih dokumenata

Student: Milan Mićović 19260

Ovaj projekat predstavlja naprednu aplikaciju za pretraživanje i analizu PDF dokumenata na srpskom jeziku. 

Sistem koristi hibridni pristup kombinujući klasični statistički algoritam BM25 i moderan AI model Multilingual-E5 za semantičko razumevanje teksta.

🛠️ Preduslovi

Za pokretanje aplikacije potrebno je da imate instaliran Python 3.9 ili noviju verziju.
Takođe, neophodno je instalirati sve potrebne biblioteke pokretanjem sledeće komande u terminalu:

pip install streamlit pymupdf classla rank_bm25 sentence-transformers numpy pandas plotly yake wordcloud matplotlib scikit-learn torch

📂 Struktura projekta

app.py - Glavni fajl aplikacije (Streamlit interfejs).

motor.py - Logika pretraživanja, indeksiranja i AI obrade.

podaci/ - Folder u koji treba ubaciti PDF dokumente za pretragu.

indeksirani_podaci.pkl - (Generiše se automatski) Baza sa sačuvanim vektorima i tekstom.

🚀 Uputstvo za pokretanje

Aplikacija se pokreće u dva jednostavna koraka:

1. Indeksiranje dokumenata
Pre prvog pokretanja pretrage, potrebno je obraditi PDF dokumente. Uverite se da se vaši PDF fajlovi nalaze u folderu podaci/, a zatim pokrenite:
python motor.py

Napomena: Prvi put će proces trajati nekoliko minuta jer sistem preuzima AI modele za srpski jezik (~500MB) i vrši lematizaciju teksta.

2. Pokretanje aplikacije (Interfejs)
Kada se indeksiranje završi i pojavi se fajl indeksirani_podaci.pkl, pokrenite Streamlit interfejs:


python -m streamlit run app.py

Aplikacija će se automatski otvoriti u vašem podrazumevanom internet pregledaču (browseru)
