import streamlit as st
import requests
from datetime import datetime
from io import BytesIO
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai
import PyPDF2
from docx import Document
import pandas as pd

# --- CONSTANTES ---
AREAS_ENADE = {
    "Ciências Sociais Aplicadas": ["Direito", "Administração", "Ciências Contábeis", "Jornalismo", "Publicidade e Propaganda", "Turismo"],
    "Engenharias": ["Engenharia de Software", "Engenharia Civil", "Engenharia de Produção", "Engenharia Elétrica", "Engenharia Mecânica"],
    "Ciências da Saúde": ["Medicina", "Enfermagem", "Farmácia", "Fisioterapia", "Nutrição"],
    "Ciências Humanas": ["Pedagogia", "História", "Letras", "Psicologia"],
}
BLOOM_LEVELS = ["Lembrar", "Compreender", "Aplicar", "Analisar", "Avaliar", "Criar"]
BLOOM_VERBS = {
    "Lembrar": ["definir", "listar", "identificar", "recordar", "nomear", "reconhecer"],
    "Compreender": ["explicar", "resumir", "interpretar", "classificar", "descrever", "discutir"],
    "Aplicar": ["usar", "implementar", "executar", "demonstrar", "resolver", "calcular"],
    "Analisar": ["diferenciar", "organizar", "atribuir", "comparar", "examinar", "categorizar"],
    "Avaliar": ["julgar", "criticar", "justificar", "avaliar", "defender", "recomendar"],
    "Criar": ["projetar", "construir", "formular", "sintetizar", "planejar", "desenvolver"]
}

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Gerador de Questões ENADE v3.4", page_icon="🎓", layout="wide")

# --- ESTADO INICIAL ---
st.session_state.setdefault("api_key", None)
st.session_state.setdefault("text_base", "")
st.session_state.setdefault("ref_final", "")
st.session_state.setdefault("search_results", [])
st.session_state.setdefault("perfil", "")
st.session_state.setdefault("competencia", "")
st.session_state.setdefault("questoes_geradas", []) 
st.session_state.setdefault("selected_index", 0)
st.session_state.setdefault("fonte_info", {})

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuração IA")
    provedor = st.selectbox("Provedor", ["OpenAI (GPT)", "Google (Gemini)"])
    st.session_state.api_key = st.text_input("Chave de API", type="password", value=st.session_state.api_key)
    if provedor.startswith("OpenAI"):
        modelo = st.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        modelo = st.selectbox("Modelo Gemini", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"])
    st.info("Versão 3.4: Busca de notícias e inclusão do texto-base corrigidas.")

    st.header("📜 Histórico da Sessão")
    if not st.session_state.questoes_geradas:
        st.caption("Nenhuma questão gerada nesta sessão.")
    else:
        titulos = [q["titulo"] for q in st.session_state.questoes_geradas]
        selected_title = st.radio(
            "Selecione uma questão para ver/editar:",
            options=titulos,
            index=st.session_state.selected_index,
            key="historico_radio"
        )
        st.session_state.selected_index = titulos.index(selected_title)

if not st.session_state.api_key:
    st.warning("Informe a chave de API na lateral para continuar.")
    st.stop()

# --- FUNÇÕES AUXILIARES ---
@st.cache_data(ttl=3600)
def extrair_conteudo_url(url: str):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string if soup.title else ""
        author_meta = soup.find("meta", attrs={"name": "author"})
        author = author_meta["content"] if author_meta else ""
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]): tag.decompose()
        return " ".join(soup.stripped_strings), title, author
    except: return None, None, None

def extrair_texto_upload(upload):
    try:
        if upload.type == "application/pdf":
            reader = PyPDF2.PdfReader(BytesIO(upload.read()))
            return "".join(p.extract_text() or "" for p in reader.pages)
        elif upload.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(BytesIO(upload.read()))
            return "\n".join(p.text for p in doc.paragraphs)
        return None
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

def search_articles(query, num=5, search_type='web'):
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt;q=0.9"}
        params = {"q": query, "hl": "pt-BR", "gl": "br", "num": num}
        if search_type == 'news':
            params['tbm'] = 'nws' 
        r = requests.get("https://www.google.com/search", headers=headers, params=params, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        # --- SELETORES ATUALIZADOS PARA BUSCA DE NOTÍCIAS ---
        for block in soup.select("div.mCBkyc, div.WlydOe, div.SoaBEf"):
            a = block.find("a", href=True)
            title_element = block.find("h3") or block.find('div', role='heading')
            if a and title_element:
                results.append({"title": title_element.get_text(), "url": a["href"]})
            if len(results) >= num: break
        return results
    except Exception as e:
        st.error(f"Erro na busca: {e}")
        return []

def chamar_llm(prompts, prov, mdl, temperature=0.7, max_tokens=2000):
    try:
        if prov.startswith("OpenAI"):
            client = OpenAI(api_key=st.session_state.api_key)
            r = client.chat.completions.create(model=mdl, messages=prompts, temperature=temperature, max_tokens=max_tokens, response_format={"type": "text"})
            return r.choices[0].message.content.strip()
        else:
            genai.configure(api_key=st.session_state.api_key)
            cfg = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens, response_mime_type="text/plain")
            m = genai.GenerativeModel(mdl)
            prompt_text = "\n".join(f"**{m['role']}**: {m['content']}" for m in prompts)
            resp = m.generate_content(prompt_text, generation_config=cfg)
            return resp.text
    except Exception as e:
        st.error(f"Erro ao chamar a API: {e}")
        return None

# --- LAYOUT PRINCIPAL ---
st.title("🎓 Gerador de Questões ENADE v3.4")
st.markdown("Bem-vindo ao gerador interativo. Siga os passos para criar, analisar e refinar suas questões.")

col_input, col_output = st.columns(2, gap="large")

with col_input:
    st.header("1. Definições da Questão")
    
    with st.container(border=True):
        st.subheader("Escopo e Tipo")
        c1, c2 = st.columns(2)
        area = c1.selectbox("Área", list(AREAS_ENADE.keys()))
        curso = c2.selectbox("Curso", AREAS_ENADE[area])
        assunto = st.text_input("Assunto central", "")
        
        tipos_questao = {
            "Múltipla Escolha Tradicional": "Enunciado com 5 alternativas (A, B, C, D, E), sendo apenas uma correta.",
            "Complementação": "Frase com uma ou mais lacunas (___) que devem ser preenchidas por uma das alternativas.",
            "Afirmação-Razão": "Duas asserções (I e II) ligadas por 'PORQUE'. O aluno avalia a veracidade de ambas e a relação entre elas.",
            "Resposta Múltipla": "Apresenta várias afirmativas (I, II, III...). O aluno deve selecionar a alternativa que indica quais estão corretas."
        }
        question_type = st.selectbox(
            "Tipo de questão", 
            options=list(tipos_questao.keys()),
            format_func=lambda k: f"{k}: {tipos_questao[k]}"
        )

    with st.container(border=True):
        st.subheader("Definição Pedagógica")
        st.session_state.perfil = st.text_input("Perfil do egresso a ser avaliado", st.session_state.perfil, help="Descreva o que se espera do profissional formado.")
        st.session_state.competencia = st.text_input("Competência a ser avaliada", st.session_state.competencia, help="Descreva a habilidade que a questão deve medir.")

    with st.container(border=True):
        st.subheader("Fonte do Texto-Base")
        opc_fonte = st.radio("Selecione a fonte:", ["Gerar com IA", "Fornecer um texto-base"], horizontal=True, key="opc_fonte")
        
        if opc_fonte == "Gerar com IA":
            if st.session_state.perfil and st.session_state.competencia and assunto:
                if st.button("Gerar Contextualização com IA", use_container_width=True):
                    with st.spinner("A IA está criando um texto-base contextualizado..."):
                        prompt_contexto = [
                            {"role": "system", "content": f"Você é um docente especialista do curso de {curso} que cria textos-base para questões do ENADE. Os textos devem ser contextualizados, apresentar uma situação-problema e ser perfeitamente alinhados às diretrizes pedagógicas."},
                            {"role": "user", "content": f"Elabore um texto-base (entre 150 e 250 palavras) para uma questão do ENADE sobre '{assunto}'. O texto deve ser uma situação-problema voltada a um egresso com o perfil: '{st.session_state.perfil}'. A avaliação focará na competência: '{st.session_state.competencia}'. O texto deve ser técnico e denso. Não inclua enunciado ou alternativas, apenas o texto-base."}
                        ]
                        tb = chamar_llm(prompt_contexto, provedor, modelo, temperature=0.6, max_tokens=400)
                        if tb:
                            st.session_state.text_base = tb
                            st.session_state.ref_final = "Texto gerado por IA."
                            st.success("Texto-base gerado!")
            else:
                st.warning("Preencha o Assunto, Perfil e Competência para gerar o texto-base.")
        else:
            tab_colar, tab_upload, tab_busca = st.tabs(["Colar Texto", "Upload de Arquivo (PDF/DOCX)", "Busca na Web"])

            with tab_colar:
                st.session_state.text_base = st.text_area("Cole o texto-base aqui:", height=150, key="tb_colar")
                st.session_state.ref_final = st.text_input("Referência ABNT do texto colado (se aplicável):", key="ref_colar")

            with tab_upload:
                up = st.file_uploader("Envie um arquivo PDF ou DOCX", type=['pdf', 'docx'])
                if up:
                    with st.spinner("Extraindo e resumindo o conteúdo do arquivo..."):
                        txt_extraido = extrair_texto_upload(up)
                        if txt_extraido:
                            prompt_resumo = [
                                {"role": "system", "content": "Você é um especialista em resumir textos para serem usados como base em questões do ENADE."},
                                {"role": "user", "content": f"Resuma o texto a seguir em um parágrafo coeso de até 200 palavras, focando nos pontos essenciais para uma questão sobre '{assunto}' no curso de {curso}.\n\nTEXTO:\n{txt_extraido[:4000]}"}
                            ]
                            st.session_state.text_base = chamar_llm(prompt_resumo, provedor, modelo, temperature=0.4)
                            st.session_state.ref_final = f"Texto adaptado de '{up.name}'."
                            st.success("Arquivo processado e resumido!")
                        else:
                            st.error("Não foi possível extrair texto do arquivo.")

            with tab_busca:
                # --- BOTÃO DE BUSCA SIMPLIFICADO ---
                if st.button("📰 Buscar Notícias", use_container_width=True, key="search_news_btn"):
                    with st.spinner(f"Buscando notícias sobre '{assunto}'..."):
                        st.session_state.search_results = search_articles(f'"{assunto}"', search_type='news')
                        if not st.session_state.search_results:
                            st.warning("Nenhuma notícia encontrada.")
                
                if st.session_state.search_results:
                    opts = [f"{r['title']}" for r in st.session_state.search_results]
                    sel_idx = st.selectbox("Selecione um resultado para usar como base:", options=range(len(opts)), format_func=lambda i: opts[i])
                    if st.button("▶️ Usar este conteúdo", key="use_search_btn"):
                        art = st.session_state.search_results[sel_idx]
                        with st.spinner(f"Extraindo e resumindo '{art['title']}'..."):
                            cont, tit, aut = extrair_conteudo_url(art["url"])
                            if cont:
                                prompt_resumo_web = [
                                    {"role": "system", "content": "Você resume conteúdos da web para serem usados como base em questões do ENADE."},
                                    {"role": "user", "content": f"Resuma o conteúdo a seguir em um parágrafo (até 200 palavras) que sirva como situação-problema sobre '{assunto}'.\n\nConteúdo:\n{cont[:4000]}"}
                                ]
                                st.session_state.text_base = chamar_llm(prompt_resumo_web, provedor, modelo, temperature=0.4)
                                st.session_state.fonte_info = {"titulo": tit, "autor": aut, "veiculo": art["url"].split("/")[2], "link": art["url"]}
                                hoje = datetime.now()
                                meses = ["jan.","fev.","mar.","abr.","mai.","jun.","jul.","ago.","set.","out.","nov.","dez."]
                                acesso = f"{hoje.day} {meses[hoje.month-1]}. {hoje.year}"
                                st.session_state.ref_final = f"Adaptado de: {aut if aut else 'Autor desconhecido'}. **{tit}**. {st.session_state.fonte_info['veiculo']}. Disponível em: <{art['url']}>. Acesso em: {acesso}."
                                st.success("Conteúdo da web processado!")
                            else:
                                st.error("Falha ao extrair conteúdo da URL.")
        st.text_area("Texto-Base a ser utilizado:", st.session_state.text_base, height=150, key="tb_final_view", disabled=True)
    
    with st.container(border=True):
        st.subheader("Parâmetros de Geração")
        with st.form("frm_gerar"):
            niv = st.select_slider("Nível Bloom", options=BLOOM_LEVELS, value="Analisar")
            dificuldade = st.slider("Nível de dificuldade", 1, 5, 3)
            gerar = st.form_submit_button("🚀 Gerar Nova Questão", use_container_width=True, type="primary")

            if gerar:
                if not st.session_state.text_base:
                    st.error("É necessário ter um Texto-Base para gerar a questão.")
                else:
                    with st.spinner("Gerando questão e análise de qualidade..."):
                        # --- PROMPT AJUSTADO PARA NÃO REPETIR O TEXTO-BASE ---
                        sys_p_geracao = """
                        Você é um docente especialista em produzir questões no estilo ENADE.
                        A partir do TEXTO-BASE e da REFERÊNCIA que serão fornecidos no prompt do usuário, sua tarefa é criar **apenas** o conteúdo da questão (ENUNCIADO, ALTERNATIVAS, GABARITO, JUSTIFICATIVAS).
                        Siga as regras:
                        - A questão deve ser inédita e alinhada à encomenda.
                        - O enunciado deve ser claro e afirmativo. Proibido pedir a 'incorreta'.
                        - Para múltipla escolha, crie 4 distratores plausíveis.
                        - **NÃO** inclua o TEXTO-BASE ou a REFERÊNCIA na sua resposta. Gere apenas o que foi pedido.
                        """
                        usr_p_geracao = f"""
                        GERAR CONTEÚDO DA QUESTÃO ENADE:
                        - Área: {area}, Curso: {curso}, Assunto: {assunto}
                        - Perfil: {st.session_state.perfil}, Competência: {st.session_state.competencia}
                        - Tipo: {question_type}, Dificuldade: {dificuldade}/5, Nível Bloom: {niv}
                        - Use o seguinte formato de saída EXATAMENTE:
                        ENUNCIADO: [Seu enunciado aqui]
                        ALTERNATIVAS:
                        A. [Alternativa A]
                        B. [Alternativa B]
                        C. [Alternativa C]
                        D. [Alternativa D]
                        E. [Alternativa E]
                        GABARITO: [Letra X]
                        JUSTIFICATIVAS:
                        A. [Justificativa para A]
                        B. [Justificativa para B]
                        C. [Justificativa para C]
                        D. [Justificativa para D]
                        E. [Justificativa para E]

                        ---
                        TEXTO-BASE PARA SUA ANÁLISE (NÃO COPIAR NA RESPOSTA):
                        {st.session_state.text_base}
                        REFERÊNCIA (NÃO COPIAR NA RESPOSTA):
                        {st.session_state.ref_final}
                        """
                        questao_parcial = chamar_llm([{"role": "system", "content": sys_p_geracao}, {"role": "user", "content": usr_p_geracao}], provedor, modelo)

                        if questao_parcial:
                            # --- MONTAGEM DA QUESTÃO FINAL COM O TEXTO-BASE ---
                            ref_formatada = f"Referência: {st.session_state.ref_final}\n\n" if st.session_state.ref_final else ""
                            questao_completa = f"TEXTO-BASE\n\n{st.session_state.text_base}\n\n{ref_formatada}{questao_parcial}"

                            sys_p_analise = """
                            Você é um avaliador de itens do ENADE, um especialista em pedagogia e avaliação. 
                            Sua tarefa é fornecer uma análise crítica e construtiva da questão fornecida.
                            Seja direto e objetivo. Use bullet points.
                            AVALIE OS SEGUINTES PONTOS:
                            - **Clareza e Pertinência:** O enunciado é claro? Ele se conecta bem ao texto-base?
                            - **Qualidade dos Distratores:** As alternativas incorretas (distratores) são plausíveis? Elas testam erros conceituais comuns ou são fáceis demais?
                            - **Alinhamento Pedagógico:** A questão realmente avalia a competência, o nível de dificuldade e o nível de Bloom solicitados?
                            - **Potencial de Melhoria:** Dê uma sugestão para melhorar a questão.
                            """
                            analise_qualidade = chamar_llm([{"role": "system", "content": sys_p_analise}, {"role": "user", "content": questao_completa}], provedor, modelo, temperature=0.3)

                            if analise_qualidade:
                                novo_item = {
                                    "titulo": f"Q{len(st.session_state.questoes_geradas) + 1}: {curso} - {assunto[:25]}...",
                                    "texto_completo": questao_completa,
                                    "analise_qualidade": analise_qualidade,
                                    "contexto": {"area": area, "curso": curso, "assunto": assunto, "perfil": st.session_state.perfil, "competencia": st.session_state.competencia, "texto_base": st.session_state.text_base}
                                }
                                st.session_state.questoes_geradas.append(novo_item)
                                st.session_state.selected_index = len(st.session_state.questoes_geradas) - 1
                                st.success("Questão e análise geradas!")
                                st.rerun()

with col_output:
    st.header("2. Análise e Refinamento")

    if not st.session_state.questoes_geradas:
        st.info("A questão gerada, junto com sua análise de qualidade e opções de refinamento, aparecerá aqui.")
    else:
        q_selecionada = st.session_state.questoes_geradas[st.session_state.selected_index]
        st.subheader(f"Visualizando: {q_selecionada['titulo']}")
        tab_view, tab_analise, tab_refino = st.tabs(["📝 Questão", "🔍 Análise de Qualidade (IA)", "✨ Refinamento Iterativo (IA)"])

        with tab_view:
            st.text_area("Texto da Questão", value=q_selecionada["texto_completo"], height=500, key=f"q_view_{st.session_state.selected_index}")
            c1, c2 = st.columns(2)
            c1.download_button("📄 Baixar esta questão (.txt)", q_selecionada["texto_completo"], f"{q_selecionada['titulo']}.txt", use_container_width=True)
            df_all = pd.DataFrame([{"titulo": q["titulo"], "questao": q["texto_completo"], "analise": q["analise_qualidade"]} for q in st.session_state.questoes_geradas])
            to_xl = BytesIO()
            df_all.to_excel(to_xl, index=False, sheet_name="Questões")
            to_xl.seek(0)
            c2.download_button("📥 Baixar todas (.xlsx)", to_xl, "banco_completo_enade.xlsx", use_container_width=True)
        with tab_analise:
            st.info("Esta análise foi gerada por uma IA especialista para ajudar na validação da questão.")
            st.markdown(q_selecionada["analise_qualidade"])
        with tab_refino:
            st.warning("Ações de refinamento modificarão a questão atual. A versão original será perdida.")
            r_c1, r_c2, r_c3 = st.columns(3)
            if r_c1.button("🤔 Tornar Mais Difícil", use_container_width=True, key=f"b_dificil_{st.session_state.selected_index}"):
                with st.spinner("Refinando para aumentar a dificuldade..."):
                    prompt_refino = f"Reescreva a questão a seguir para torná-la significativamente mais difícil, mantendo o mesmo gabarito.\n\nQUESTÃO ATUAL:\n{q_selecionada['texto_completo']}"
                    texto_refinado = chamar_llm([{"role": "user", "content": prompt_refino}], provedor, modelo)
                    st.session_state.questoes_geradas[st.session_state.selected_index]["texto_completo"] = texto_refinado
                    st.rerun()
            if r_c2.button("✍️ Simplificar o Enunciado", use_container_width=True, key=f"b_simplificar_{st.session_state.selected_index}"):
                with st.spinner("Refinando para simplificar o enunciado..."):
                     prompt_refino = f"Reescreva apenas o ENUNCIADO da questão a seguir para torná-lo mais claro e direto, sem alterar o gabarito.\n\nQUESTÃO ATUAL:\n{q_selecionada['texto_completo']}"
                     texto_refinado = chamar_llm([{"role": "user", "content": prompt_refino}], provedor, modelo)
                     st.session_state.questoes_geradas[st.session_state.selected_index]["texto_completo"] = texto_refinado
                     st.rerun()
            if r_c3.button("🔄 Regenerar Alternativas", use_container_width=True, key=f"b_alternativas_{st.session_state.selected_index}"):
                with st.spinner("Regenerando as alternativas..."):
                    prompt_refino = f"Mantenha o TEXTO-BASE e o ENUNCIADO da questão a seguir, mas gere um conjunto completamente novo de 5 ALTERNATIVAS, GABARITO e suas respectivas JUSTIFICATIVAS.\n\nQUESTÃO ATUAL:\n{q_selecionada['texto_completo']}"
                    texto_refinado = chamar_llm([{"role": "user", "content": prompt_refino}], provedor, modelo)
                    st.session_state.questoes_geradas[st.session_state.selected_index]["texto_completo"] = texto_refinado
                    st.rerun()

# Botão para limpar a sessão
if st.sidebar.button("🔴 Encerrar e Limpar Sessão", use_container_width=True):
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    st.rerun()
