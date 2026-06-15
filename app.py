import streamlit as st
import pickle
import os
import fitz  
import base64
import pandas as pd
import plotly.express as px
from motor import generisi_mapu_dokumenata, generisi_rezime, hibridna_pretraga, nadji_slicne_stranice, podaci, prosiri_upit

st.set_page_config(page_title="Hibridna pretraga akademskih dokumenata", layout="wide")

def display_pdf(file_path, page_num):
    if not os.path.exists(file_path):
        st.error(f"Fajl nije nađen na putanji: {file_path}")
        return

    try:
        doc = fitz.open(file_path)
        new_doc = fitz.open()
        
        idx = page_num - 1
        if 0 <= idx < len(doc):
            new_doc.insert_pdf(doc, from_page=idx, to_page=idx)
            pdf_bytes = new_doc.write()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            
            pdf_display = f'''
                <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" height="800" type="application/pdf"></iframe>
            '''
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Preuzmi ceo dokument",
                data=open(file_path, "rb"),
                file_name=os.path.basename(file_path),
                mime="application/pdf"
            )
        else:
            st.error(f"Greška: Dokument nema stranu {page_num}")
        doc.close()
        new_doc.close()
    except Exception as e:
        st.error(f"Greška pri učitavanju PDF-a: {e}")

st.title("🔍 Hibridni Pretraživač akademskih dokumenata")
st.subheader("Milan Mićović 19260 - Projekat iz Pretraživanja informacija")
st.markdown("---")

@st.cache_resource
def ucitaj_bazu():
    if os.path.exists("indeksirani_podaci.pkl"):
        with open("indeksirani_podaci.pkl", "rb") as f:
            return pickle.load(f)
    return None

baza = ucitaj_bazu()

if baza:
    # Sidebar
    st.sidebar.header("📊 Statistika sistema")
    
    df_baza = pd.DataFrame(baza)
    
    stats = df_baza.groupby('ime').size().reset_index(name='broj_strana')
    
    st.sidebar.metric("Ukupno dokumenata", len(stats))
    st.sidebar.metric("Ukupno stranica", len(baza))
    
    st.sidebar.write("---")
    st.sidebar.write("**Detalji po dokumentu:**")
    
    for index, row in stats.iterrows():
        st.sidebar.markdown(f"📄 **{row['ime']}**")
        st.sidebar.caption(f"Broj indeksiranih stranica: {row['broj_strana']}")
        st.sidebar.write("")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Pretraga", "📉 Evaluacija", "🗺️ Mapa Znanja", "⚖️ Upoređivanje"])

    #TAB 1:
    with tab1:
        dostupni_fajlovi = list(df_baza['ime'].unique())
        izabrani_fajlovi = st.multiselect("Traži samo u dokumentima:", dostupni_fajlovi, default=dostupni_fajlovi)

        query = st.text_input("Unesite istraživačko pitanje:")
        
        if query:
            predlozi = prosiri_upit(query, baza)
            filtrirana_baza = [d for d in baza if d['ime'] in izabrani_fajlovi]
            if not filtrirana_baza:
                st.warning("⚠️ Molimo izaberite bar jedan dokument za pretragu.")
            else:
                predlozi = prosiri_upit(query, filtrirana_baza) 
                if predlozi:
                    st.write(f"💡 *Možda vas zanima i:* {', '.join([f'**{p}**' for p in predlozi])}")
            
            results = hibridna_pretraga(query,filtrirana_baza)
            
            for r in results:
                with st.container():
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader(f"📄 {r['ime']} (Strana {r['strana']})")
                        st.markdown(r['isečak'])
                        if st.button(f"📋 Podaci", key=f"cit_{r['ime']}_{r['strana']}"):
                            citat = podaci(r)
                            st.code(citat, language="text")
                            st.toast("Uspesno pronadjeni podaci!")
                        
                        c1, c2, c3 = st.columns(3)
                        if c1.button(f"👁️ Pregled", key=f"p_{r['ime']}_{r['strana']}"):
                            display_pdf(r['putanja'], r['strana'])
                        
                        if c2.button(f"🤖 Rezime", key=f"r_{r['ime']}_{r['strana']}"):
                            pun_tekst = next(d['original'] for d in baza if d['ime'] == r['ime'] and d['strana'] == r['strana'])
                            with st.spinner("..."):
                                rezime = generisi_rezime(pun_tekst)
                                st.success(f"**Rezime:** {rezime}")
                        
                        if c3.button(f"🔗 Slično", key=f"s_{r['ime']}_{r['strana']}"):
                            curr_emb = next(d['embedding'] for d in baza if d['ime'] == r['ime'] and d['strana'] == r['strana'])
                            slicni = nadji_slicne_stranice(curr_emb, baza)
                            for s in slicni:
                                st.write(f"📍 {s['ime']} (str. {s['strana']}) - Sličnost: {s['score']:.2f}")
                    
                    with col2:
                        st.metric("Score", f"{r['score']:.4f}")
                        st.bar_chart({"BM25": r['bm25_part'], "AI": r['sem_part']})
                    st.markdown("---")

    with tab2:
        st.header("🔬 Evaluacija Sistema")
        st.write("""
        Ovaj modul vrši automatsko testiranje sistema koristeći **'Ground Truth'** skup podataka 
        zasnovan na indeksiranim akademskim radovima.
        """)

        test_pitanja = [
            {"upit": "mašinsko učenje u spektroskopiji plazme", "tacan_doc": "Fizicki.pdf"},
            {"upit": "genetski polimorfizmi i farmakokinetika takrolimusa", "tacan_doc": "Medicina.pdf"},
            {"upit": "metabolomika u kontroli kvaliteta lekovitog bilja", "tacan_doc": "Hemija.pdf"},
            {"upit": "pravni okvir za digitalnu imovinu", "tacan_doc": "Pravo.pdf"} 
        ]

        if st.button("🚀 Pokreni Test"):
            rezultati_testa = []
            hit_at_1 = 0
            hit_at_3 = 0

            with st.spinner("Evaluacija je u toku..."):
                for test in test_pitanja:

                    pronadjeni = hibridna_pretraga(test['upit'], baza, top_n=3)
                    
                    top_1_ime = pronadjeni[0]['ime']
                    svi_pronadjeni_imena = [p['ime'] for p in pronadjeni]

                    is_p1 = top_1_ime.lower() == test['tacan_doc'].lower()
                    if is_p1:
                        hit_at_1 += 1
                    
                    is_top3 = any(test['tacan_doc'].lower() == ime.lower() for ime in svi_pronadjeni_imena)
                    if is_top3:
                        hit_at_3 += 1
                    
                    rezultati_testa.append({
                        "Istraživački upit": test['upit'],
                        "Očekivani rad": test['tacan_doc'],
                        "Sistem pronašao (Top 1)": top_1_ime,
                        "Status": "✅ HIT" if is_p1 else "❌ MISS",
                        "U Top 3": "Da" if is_top3 else "Ne"
                    })

            p_at_1 = hit_at_1 / len(test_pitanja)
            recall_at_3 = hit_at_3 / len(test_pitanja) 
            f1_score = 0 if (p_at_1 + recall_at_3) == 0 else (2 * p_at_1 * recall_at_3) / (p_at_1 + recall_at_3)

            m1, m2, m3 = st.columns(3)
            m1.metric("Precision @ 1", f"{p_at_1:.2f}", help="Koliko često je prvi rezultat onaj pravi.")
            m2.metric("Recall @ 3", f"{recall_at_3:.2f}", help="Koliko često je pravi rad bar među prva tri.")
            m3.metric("F1 Score", f"{f1_score:.2f}")

            st.table(rezultati_testa)

            st.info(f"""
            **Analiza:** Sistem je postigao preciznost od {p_at_1*100:.0f}% na prvom mestu. 
            Ovo dokazuje da hibridni model (BM25 + E5) uspešno mapira semantiku pitanja na 
            specifičan naučni sadržaj.
            """)

    with tab3:
        st.header("🗺️ Mapa Znanja (Semantički prostor)")
        st.write("Grafikon prikazuje kako AI grupiše stranice prema sličnosti sadržaja.")

        with st.spinner("Generisanje mape..."):
            mapa_df = generisi_mapu_dokumenata(baza)
            fig = px.scatter(mapa_df, x="x", y="y", 
                        color="Dokument", 
                        hover_data=["Strana", "Tagovi"],
                        title="Semantička sličnost stranica u dokumentima")
        
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("---")
        st.subheader("☁️ Ključni koncepti u celoj bazi")

        srpske_stop_reci = {
            "i", "a", "ali", "pa", "te", "da", "u", "na", "po", "sa", "od", "do", "iz", 
            "za", "kod", "kao", "kroz", "preko", "ili", "niti", "niti", "jer", "ako", 
            "dok", "tako", "tada", "gde", "koji", "koja", "koje", "čiji", "ova", "ovo", 
            "taj", "one", "sve", "svih", "ne", "nije", "bi", "bio", "bila", "biti", 
            "jesu", "smo", "sam", "su", "član", "stav", "broj", "ovog", "onog", "ili",
            "takođe", "može", "mogu", "ovim", "toga", "reč", "reči", "strana", "godina",
            "tabela", "slika", "grafikon", "naslov", "uvod", "zaključak", "disertacija",
            "univerzitet", "fakultet", "beograd", "niš", "novi sad", "rad", "tema",
            "ispod", "iznad", "prema", "nakon", "tokom", "putem", "kome", "čemu", "čega",
            "или","али","који","није","већ","како","као","and","doktorska","београд"
        }

        imena_autora = set()
        for d in baza:
            pun_autor = d.get('autor', '')
            for rec in pun_autor.split():
                cista_rec = rec.strip(",.").lower()
                if len(cista_rec) > 2:
                    imena_autora.add(cista_rec)
        
        finalne_stop_reci = srpske_stop_reci.union(imena_autora)

        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        svi_tagovi = " ".join([" ".join(d.get('kljucne_reci', [])) for d in baza])
        
        wc = WordCloud(
            width=800, 
            height=400, 
            background_color="black", 
            colormap="Blues",
            stopwords=finalne_stop_reci, 
            collocations=False 
        ).generate(svi_tagovi.lower()) 
        
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    
    with tab4:
        st.header("⚖️ Semantičko poređenje dokumenata")
        st.write("Analiza tematskog preklapanja korišćenjem AI vektora i ekstrakcije ključnih pojmova.")
        
        fajlovi = list(df_baza['ime'].unique())
        col_a, col_b = st.columns(2)
        
        with col_a:
            doc1 = st.selectbox("Prvi dokument:", fajlovi, key="d1")
        with col_b:
            doc2 = st.selectbox("Drugi dokument:", fajlovi, key="d2")
            
        if st.button("Uporedi radove"):
            from motor import uporedi_dokumente
            procenat, zajednicke = uporedi_dokumente(doc1, doc2, baza)
            
            st.write(f"### Sličnost: {procenat:.2f}%")
            st.progress(min(procenat / 100, 1.0))
            
            if zajednicke:
                st.write(f"**Zajedničke teme:** {', '.join([f'`{z}`' for z in zajednicke])}")
            else:
                st.write("**Zajedničke teme:** Nema značajnog preklapanja u ključnim pojmovima.")
            
            if procenat > 70:
                st.success("Visoka tematska podudarnost.")
            elif procenat > 30:
                st.warning("Umerena sličnost (moguće dodirne tačke u metodologiji).")
            else:
                st.error("Radovi su iz potpuno različitih oblasti.")

else:
    st.error("Baza nije pronađena! Prvo pokrenite motor.py da indeksirate PDF dokumente.")