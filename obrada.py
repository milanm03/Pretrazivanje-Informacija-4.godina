import fitz  
import os
import classla

# Inicijalizacija Classla za srpski jezik
print("Učitavanje modela za srpski jezik...")
classla.download('sr')
nlp = classla.Pipeline('sr', processors='tokenize,pos,lemma')

def ucitaj_i_lematizuj_pdfove(folder_putanja):
    dokumenti = []
    
    for fajl in os.listdir(folder_putanja):
        if fajl.endswith(".pdf"):
            putanja = os.path.join(folder_putanja, fajl)
            print(f"Obrađujem: {fajl}")
            
            doc = fitz.open(putanja)
            pun_tekst = ""
            for strana in doc:
                pun_tekst += strana.get_text()
            
            # Lematizacija
            obradjen_doc = nlp(pun_tekst)
            lematizovan_tekst = " ".join([word.lemma.lower() for sent in obradjen_doc.sentences for word in sent.words])
            
            dokumenti.append({
                "ime_fajla": fajl,
                "originalni_tekst": pun_tekst,
                "lematizovan_tekst": lematizovan_tekst
            })
    return dokumenti

