import os
import fitz  
import classla
import pickle
import numpy as np
import torch
import yake
import re
from sklearn.decomposition import PCA
import plotly.express as px
from sentence_transformers import CrossEncoder
from transformers import pipeline
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


FOLDER_DOKUMENTI = "podaci"
INDEX_FAJL = "indeksirani_podaci.pkl"  # Ovde čuvamo obrađene podatke da ne čekamo svaki put
kw_extractor = yake.KeywordExtractor(lan="sr", n=1, dedupLim=0.9, top=5, features=None)
print("Učitavam modele (ovo može potrajati prvi put)...")
classla.download('sr')
nlp = classla.Pipeline('sr', processors='tokenize,pos,lemma')
model_semantika = SentenceTransformer('intfloat/multilingual-e5-small')

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def generisi_mapu_dokumenata(baza):
    import pandas as pd
    from sklearn.decomposition import PCA
    
    embeddings = np.array([d['embedding'].cpu().numpy() for d in baza])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Dokument": [d['ime'] for d in baza],
        "Strana": [d['strana'] for d in baza],
        "Tagovi": [", ".join(d.get('kljucne_reci', [])) for d in baza]
    })
    return df

def napredno_rerangiranje(query, rezultati):
    parovi = [[query, r['isečak']] for r in rezultati]
    scores = reranker.predict(parovi)
    
    for i in range(len(rezultati)):
        rezultati[i]['cross_score'] = float(scores[i])
        
    return sorted(rezultati, key=lambda x: x['cross_score'], reverse=True)

def lematizuj_tekst(tekst):
    doc = nlp(tekst)
    return " ".join([word.lemma.lower() for sent in doc.sentences for word in sent.words if word.lemma])

def izvuci_podatke(putanja):
    import re
    import fitz
    import os

    doc = fitz.open(putanja)
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    
    spans = []
    for b in blocks:
        if "lines" in b:
            for l in b["lines"]:
                for s in l["spans"]:
                    txt = s["text"].strip()
                    if len(txt) > 2:
                        spans.append({
                            "text": txt,
                            "size": s["size"],
                            "y": s["origin"][1]
                        })
    
    if not spans:
        doc.close()
        return os.path.basename(putanja), "Nepoznat autor", "n.d."

    admin_reci = ["univerzitet", "fakultet", "akademija", "beograd", "niš", "novi sad", "disertacija", "teza"] #stop reci 
    naslov_kandidati = [s for s in spans if not any(w in s["text"].lower() for w in admin_reci)]
    
    if naslov_kandidati:
        max_font = max(naslov_kandidati, key=lambda x: x["size"])["size"]
        naslov_spans = [s for s in spans if abs(s["size"] - max_font) < 0.5]
        naslov = " ".join([s["text"] for s in naslov_spans])
    else:
        naslov = os.path.basename(putanja)

    mentori_titule = ["dr", "prof", "doc", "mr", "mentor", "komisija", "član", "ispitna", "kandidat", "student", "ime i prezime"] #stop reci
    
    kandidati_autor = []
    for s in spans:
        txt = s["text"].strip()
        txt_lower = txt.lower()
        
        # Preskačemo ako je to naslov
        if txt in naslov: continue
        # Preskačemo ako sadrži admin reči ili titule mentora
        if any(w in txt_lower for w in admin_reci): continue
        if any(m in txt_lower for m in mentori_titule): continue
        
        # Proveravamo da li liči na ime
        ciste_reci = re.sub(r'[^\w\s]', '', txt).split()
        if 2 <= len(ciste_reci) <= 4:
            if all(w[0].isupper() for w in ciste_reci if w):
                kandidati_autor.append(s)

    if kandidati_autor:
        najbolji_span = max(kandidati_autor, key=lambda x: x["size"])
        autor = najbolji_span["text"].strip()
        autor = re.sub(r'^(kandidat|student|ime i prezime)[:\s]+', '', autor, flags=re.IGNORECASE)
    else:
        autor = "Nepoznat autor"

    godina_match = re.search(r'\b(20\d{2}|19\d{2})\b', page.get_text())
    godina = godina_match.group(0) if godina_match else "n.d."

    doc.close()
    return naslov.strip(), autor.strip(), godina

def kreiraj_indeks():
    podaci_za_indeks = []
    if not os.path.exists(FOLDER_DOKUMENTI):
        os.makedirs(FOLDER_DOKUMENTI)

    for fajl in os.listdir(FOLDER_DOKUMENTI):
        if fajl.endswith(".pdf"):
            putanja_fajla = os.path.join(FOLDER_DOKUMENTI, fajl) 
            print(f"Indeksiram: {fajl}...")
            
            doc = fitz.open(putanja_fajla)
            for page_num, page in enumerate(doc):
                tekst_strane = page.get_text().strip()
                if len(tekst_strane) < 100: continue
                
                lematizovan = lematizuj_tekst(tekst_strane)
                embedding = model_semantika.encode(f"passage: {tekst_strane}", convert_to_tensor=True)
                
                keywords = kw_extractor.extract_keywords(tekst_strane)
                kljucne_reci = [kw[0] for kw in keywords]
                naslov, autor, god = izvuci_podatke(putanja_fajla)
                
                podaci_za_indeks.append({
                    "ime": fajl,
                    "putanja": putanja_fajla,
                    "strana": page_num + 1,
                    "original": tekst_strane,
                    "lematizovan": lematizovan,
                    "embedding": embedding,
                    "kljucne_reci": kljucne_reci,
                    "naslov": naslov, 
                    "autor": autor,
                    "godina": god
                })
            doc.close()
            
    with open(INDEX_FAJL, "wb") as f:
        pickle.dump(podaci_za_indeks, f)
    print("Baza je uspešno kreirana sa putanjama!")
    return podaci_za_indeks

def hibridna_pretraga(upit, podaci, top_n=7):
    upit_lematizovan = lematizuj_tekst(upit)
    upit_embedding = model_semantika.encode(f"query: {upit}", convert_to_tensor=True)
    
    corpus_lematizovan = [d['lematizovan'].split() for d in podaci]
    bm25 = BM25Okapi(corpus_lematizovan)
    bm25_scores = bm25.get_scores(upit_lematizovan.split())
    # Normalizacija BM25 (da bude između 0 i 1)
    if max(bm25_scores) != 0:
        bm25_scores = bm25_scores / max(bm25_scores)
    
    
    import torch
    doc_embeddings_tensor = torch.stack([d['embedding'] for d in podaci])
    semantika_scores = util.cos_sim(upit_embedding, doc_embeddings_tensor)[0].cpu().numpy()
    
    finalni_rezultati = []
    for i in range(len(podaci)):
        finalni_rezultati = []
    for i in range(len(podaci)):
        score = (bm25_scores[i] * 0.3) + (semantika_scores[i].item() * 0.7)
        finalni_rezultati.append({
            "ime": podaci[i]['ime'],
            "putanja": podaci[i].get('putanja', ""), 
            "strana": podaci[i].get('strana', 1),
            "original": podaci[i]['original'],
            "score": score,
            "bm25_part": float(bm25_scores[i]),
            "sem_part": float(semantika_scores[i]),
            "kljucne_reci": podaci[i].get('kljucne_reci', []),
            "naslov": podaci[i].get("naslov","Nepoznat naslov"),
            "autor":podaci[i].get("autor","Nepoznat autor"),
            "godina":podaci[i].get("godina","n.d")

        })
    
    finalni_rezultati = sorted(finalni_rezultati, key=lambda x: x['score'], reverse=True)
    top_rezultati = finalni_rezultati[:top_n]
    
    for r in top_rezultati:
        r['isečak'] = pronadji_najbolji_isecak(upit, r['original'])
        
    return top_rezultati
    

def pronadji_najbolji_isecak(upit, pun_tekst, prozor_reci=50):
    #Priprema reci iz upita
    reci_upita = [w.lower() for w in upit.split() if len(w) > 3]
    if not reci_upita:
        return pun_tekst[:300] + "..."

    #Podela teksta na recenice
    recenice = re.split(r'(?<=[.!?]) +', pun_tekst.replace('\n', ' '))
    
    najbolja_recenica = ""
    max_score = -1

    #Pronalazenje recenice koja ima najvise slicnosti sa upitom
    for i, recenica in enumerate(recenice):
        score = sum(1 for rec in reci_upita if rec in recenica.lower())
        
        if score > max_score:
            max_score = score
            start = max(0, i-1)
            end = min(len(recenice), i+2)
            najbolja_recenica = " ".join(recenice[start:end])

    if max_score <= 0:
        return pun_tekst[:300] + "..."

    finalni_isecak = najbolja_recenica
    for rec in reci_upita:
        pattern = re.compile(re.escape(rec), re.IGNORECASE)
        finalni_isecak = pattern.sub(f"**{rec.upper()}**", finalni_isecak)

    return f"... {finalni_isecak.strip()} ..."

if __name__ == "__main__":

    if not os.path.exists(INDEX_FAJL):
        baza = kreiraj_indeks()
    else:
        print("Učitavam postojeći indeks...")
        with open(INDEX_FAJL, "rb") as f:
            baza = pickle.load(f)
            
    test_upit = input("Unesite upit za pretragu: ")
    rez = hibridna_pretraga(test_upit, baza)
    
    print("\nREZULTATI PRETRAGE:\n")
    for r in rez:
        print(f"\nDokument: {r['ime']}")
        print(f"Isečak: {r['isečak']}")
        print("-" * 30)

def generisi_rezime(tekst, broj_recenica=3):
    if not tekst or len(tekst) < 100:
        return "Tekst je prekratak za rezime."

    #Podela teksta na rečenice
    import re
    recenice = re.split(r'(?<=[.!?]) +', tekst.replace('\n', ' '))
    if len(recenice) <= broj_recenica:
        return tekst

    #Pretvaramo recenice u vektore
    recenice_embs = model_semantika.encode(recenice, convert_to_tensor=True)
    
    #Nalazimo prosecan vektor
    import torch
    centroid = torch.mean(recenice_embs, dim=0, keepdim=True)
    
    #Nalazimo recenice koje su najslicnije
    kosna_slicnost = util.cos_sim(centroid, recenice_embs)[0]
    top_indices = torch.topk(kosna_slicnost, k=min(broj_recenica, len(recenice))).indices
    
    #Vracamo te recenice u redosledu koji je u tekstu
    finalni_rezime = [recenice[idx] for idx in sorted(top_indices.tolist())]
    return " ".join(finalni_rezime)

def nadji_slicne_stranice(strana_embedding, baza, top_k=3):

    import torch
    sve_strane_embs = torch.stack([d['embedding'] for d in baza])
    scores = util.cos_sim(strana_embedding, sve_strane_embs)[0].cpu().numpy()
    
    top_indices = np.argsort(scores)[::-1][1:top_k+1]
    
    slicne = []
    for idx in top_indices:
        slicne.append({
            "ime": baza[idx]['ime'],
            "strana": baza[idx]['strana'],
            "score": scores[idx]
        })
    return slicne

def prosiri_upit(query, baza, top_n=3):
    
    query_emb = model_semantika.encode(query, convert_to_tensor=True)
    
    sve_reci = {}
    for d in baza:
        for kw in d.get('kljucne_reci', []):
            if kw.lower() != query.lower():
                sve_reci[kw] = sve_reci.get(kw, 0) + 1
    
    jedinstvene_reci = list(sve_reci.keys())
    if not jedinstvene_reci: return []
    
    reci_embs = model_semantika.encode(jedinstvene_reci, convert_to_tensor=True)
    slicnost = util.cos_sim(query_emb, reci_embs)[0]
    
    top_indices = torch.topk(slicnost, k=min(top_n, len(jedinstvene_reci))).indices
    predlozi = [jedinstvene_reci[idx] for idx in top_indices]
    return predlozi

def podaci(doc_info):

    naslov = doc_info.get('naslov', 'Nepoznat naslov')
    autor = doc_info.get('autor', 'Nepoznat autor')
    godina = doc_info.get('godina', 'Nepoznata godina')
    
    return f"Naziv: {naslov}\nAutor: {autor}\nGodina: {godina}"

def uporedi_dokumente(ime1, ime2, baza):
    import torch
    
    
    emb1 = torch.stack([d['embedding'] for d in baza if d['ime'] == ime1])
    emb2 = torch.stack([d['embedding'] for d in baza if d['ime'] == ime2])
    
    cent1 = torch.mean(emb1, dim=0, keepdim=True)
    cent2 = torch.mean(emb2, dim=0, keepdim=True)
    
    semanticka_slicnost = util.cos_sim(cent1, cent2).item()
    sem_score = max(0, (semanticka_slicnost - 0.7) / 0.3) 

    tagovi1 = set()
    for d in baza:
        if d['ime'] == ime1:
            tagovi1.update(d.get('kljucne_reci', []))
            
    tagovi2 = set()
    for d in baza:
        if d['ime'] == ime2:
            tagovi2.update(d.get('kljucne_reci', []))
    
    if not tagovi1 or not tagovi2:
        leksicka_slicnost = 0
    else:
        presek = tagovi1.intersection(tagovi2)
        unija = tagovi1.union(tagovi2)
        leksicka_slicnost = len(presek) / len(unija)

    finalni_score = (sem_score * 0.4) + (leksicka_slicnost * 0.6)
    
    zajednicke = list(tagovi1.intersection(tagovi2))
    return finalni_score * 100, zajednicke

