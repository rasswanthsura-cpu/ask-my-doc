import streamlit as st
import streamlit.components.v1 as components
import PyPDF2
import faiss
import numpy as np
import ollama

st.set_page_config(page_title="AI Doc Chat", layout="wide")

# 🎨 Custom CSS for Futuristic UI & Transparent Backgrounds
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

/* Make Streamlit background transparent so particles show through */
.stAppHeader, .stApp {
    background: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background: transparent !important;
    font-family: 'Outfit', sans-serif;
    color: #e0e0e0;
}

/* Glassmorphism Upload Container */
.upload-container {
    background: rgba(15, 15, 25, 0.4) !important;
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid rgba(0, 255, 255, 0.2);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 0 20px rgba(0,0,0,0.5);
}

/* Buttons - Glowing Neon Effect */
.stButton>button {
    background: rgba(20, 20, 30, 0.8) !important;
    border: 1px solid rgba(0, 243, 255, 0.3) !important;
    color: #00f3ff !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 10px rgba(0, 243, 255, 0.1) !important;
    font-weight: 600 !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
}

.stButton>button:hover {
    background: rgba(0, 243, 255, 0.1) !important;
    border-color: #00f3ff !important;
    box-shadow: 0 0 20px rgba(0, 243, 255, 0.5), 0 0 40px rgba(0, 243, 255, 0.2) !important;
    transform: translateY(-2px);
}

/* Minimal styling for Streamlit's native dropzone */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.3s ease;
}

[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploadDropzone"]:hover {
    border-color: #00f3ff !important;
}

/* 1. Hide extraneous file uploader text elements natively */
[data-testid="stFileUploaderDropzone"] div[data-testid="stText"],
[data-testid="stFileUploadDropzone"] div[data-testid="stText"],
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploadDropzone"] svg,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploadDropzone"] button span {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
}

/* 2. Shape the native button into a sleek, wordless icon circle */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploadDropzone"] button {
    /* Mask the native text */
    color: transparent !important;
    font-size: 0px !important; 
    
    /* Perfect Circle Setup */
    width: 65px !important;
    height: 65px !important;
    border-radius: 50% !important;
    background: rgba(20, 20, 30, 0.8) !important;
    border: 1px solid rgba(0, 243, 255, 0.5) !important;
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.15) !important;
    
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    min-width: unset !important;
    position: relative;
    overflow: hidden !important;
}

/* 3. Inject the clean SVG arrow directly in the center WITHOUT using string content */
[data-testid="stFileUploaderDropzone"] button::after,
[data-testid="stFileUploadDropzone"] button::after {
    content: "" !important;
    display: block !important;
    width: 25px !important;
    height: 25px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2300f3ff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E");
    background-size: contain !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    visibility: visible !important;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

/* 4. Hover glow */
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploadDropzone"] button:hover {
    background: rgba(0, 243, 255, 0.1) !important;
    box-shadow: 0 0 25px rgba(0, 243, 255, 0.5) !important;
}


/* Chat Messages - Glassmorphic */
[data-testid="stChatMessage"] {
    background: rgba(25, 25, 35, 0.5) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

/* User Messages get a subtle purple tint */
[data-testid="stChatMessage"]:has(> div > div > [data-testid="stChatMessageAvatarUser"]) {
    border-left: 2px solid #ab00ff;
    background: rgba(171, 0, 255, 0.03) !important;
}

/* Assistant Messages get a subtle cyan tint */
[data-testid="stChatMessage"]:has(> div > div > [data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 2px solid #00f3ff;
    background: rgba(0, 243, 255, 0.03) !important;
}

/* For older Streamlit versions fallback */
[data-testid="stChatMessage"]:nth-child(even) {
    background: rgba(0, 243, 255, 0.03) !important;
    border-left: 2px solid #00f3ff;
}
[data-testid="stChatMessage"]:nth-child(odd) {
    background: rgba(171, 0, 255, 0.03) !important;
    border-left: 2px solid #ab00ff;
}

/* Chat Input Bar */
.stChatInputContainer {
    background: rgba(15, 15, 20, 0.8) !important;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(0, 243, 255, 0.2) !important;
    border-radius: 15px !important;
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.5) !important;
}

.stChatInputContainer:focus-within {
    border-color: #00f3ff !important;
    box-shadow: 0 0 20px rgba(0, 243, 255, 0.3) !important;
}

/* Text overrides */
h1, h2, h3, p, div, span {
    font-family: 'Outfit', sans-serif !important;
}

/* Title gradient */
.title-gradient {
    background: -webkit-linear-gradient(45deg, #00f3ff, #ab00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# 💫 Interactive Document & Data Flow Background
components.html("""
<script>
    const doc = window.parent.document;
    if (!doc.getElementById("ai-doc-bg")) {
        // Remove old particles if present
        const oldP = doc.getElementById("particles-js");
        if (oldP) oldP.remove();

        const bg = doc.createElement("div");
        bg.id = "ai-doc-bg";
        bg.style.position = "fixed";
        bg.style.top = "0";
        bg.style.left = "0";
        bg.style.width = "100vw";
        bg.style.height = "100vh";
        bg.style.zIndex = "-1";
        bg.style.overflow = "hidden";
        bg.style.background = "radial-gradient(circle at 50% 50%, #0a0a1a 0%, #050510 100%)";
        doc.body.insertBefore(bg, doc.body.firstChild);
        
        const style = doc.createElement("style");
        style.innerHTML = `
            .floating-doc {
                position: absolute;
                pointer-events: none;
                animation: floatUp linear infinite;
                filter: drop-shadow(0 0 8px currentColor);
            }
            @keyframes floatUp {
                0% { transform: translateY(110vh) rotate(-10deg) scale(0.8); opacity: 0; }
                10% { opacity: 0.6; }
                90% { opacity: 0.6; }
                100% { transform: translateY(-10vh) rotate(10deg) scale(1.1); opacity: 0; }
            }
            .neural-line {
                position: absolute;
                width: 1px;
                background: linear-gradient(to bottom, transparent, rgba(0, 243, 255, 0.5), transparent);
                animation: dataFlow linear infinite;
            }
            @keyframes dataFlow {
                0% { transform: translateY(-100vh); }
                100% { transform: translateY(100vh); }
            }
        `;
        doc.head.appendChild(style);

        const svgs = [
            // PDF Document
            `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>`,
            // Text Lines
            `<svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="6" x2="20" y2="6"></line><line x1="4" y1="12" x2="14" y2="12"></line><line x1="4" y1="18" x2="18" y2="18"></line></svg>`,
            // Upload Arrow / File
            `<svg width="35" height="35" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>`,
            // Data Block / Neural node
            `<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>`
        ];

        // Create floating document SVGs
        for(let i = 0; i < 20; i++) {
            let el = doc.createElement("div");
            el.className = "floating-doc";
            el.innerHTML = svgs[Math.floor(Math.random() * svgs.length)];
            el.style.left = Math.random() * 100 + "vw";
            el.style.animationDuration = (Math.random() * 20 + 20) + "s";
            el.style.animationDelay = "-" + (Math.random() * 20) + "s";
            el.style.color = Math.random() > 0.5 ? "rgba(0, 243, 255, 0.15)" : "rgba(171, 0, 255, 0.15)";
            bg.appendChild(el);
        }

        // Create neural data flow lines
        for(let i = 0; i < 15; i++) {
            let line = doc.createElement("div");
            line.className = "neural-line";
            line.style.left = Math.random() * 100 + "vw";
            line.style.height = (Math.random() * 40 + 20) + "vh";
            line.style.animationDuration = (Math.random() * 8 + 4) + "s";
            line.style.animationDelay = "-" + (Math.random() * 10) + "s";
            bg.appendChild(line);
        }
        
        // Load lightweight canvas logic for abstract connecting dots (Neural network map)
        const canvas = doc.createElement("canvas");
        canvas.style.position = "absolute";
        canvas.style.top = "0";
        canvas.style.left = "0";
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        canvas.style.opacity = "0.5";
        bg.appendChild(canvas);
        
        const ctx = canvas.getContext("2d");
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;
        
        window.parent.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        const dots = [];
        for(let i=0; i<60; i++) {
            dots.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 2 + 1
            });
        }

        let mouse = {x: -1000, y: -1000};
        window.parent.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });

        function draw() {
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = "#00f3ff";
            ctx.strokeStyle = "rgba(171, 0, 255, 0.15)";
            
            for(let i=0; i<dots.length; i++) {
                let p = dots[i];
                p.x += p.vx;
                p.y += p.vy;
                
                if(p.x < 0 || p.x > width) p.vx *= -1;
                if(p.y < 0 || p.y > height) p.vy *= -1;
                
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fill();
                
                for(let j=i+1; j<dots.length; j++) {
                    let p2 = dots[j];
                    let dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                    if(dist < 120) {
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                }
                
                // Repel mouse
                let mDist = Math.hypot(p.x - mouse.x, p.y - mouse.y);
                if(mDist < 100) {
                    p.x += (p.x - mouse.x) * 0.01;
                    p.y += (p.y - mouse.y) * 0.01;
                }
            }
            requestAnimationFrame(draw);
        }
        draw();
    }
</script>
""", height=0, width=0)

# 🧠 Read PDF
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ✂️ Split text
def make_chunks(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# 🔢 Embeddings (Local via Ollama)
def get_embeddings(chunks):
    try:
        response = ollama.embed(
            model='nomic-embed-text',
            input=chunks
        )
        return np.array(response['embeddings']).astype("float32")
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return np.array([])

# 💾 FAISS
def build_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# 🔍 Search (Local via Ollama)
def search(question, chunks, index):
    try:
        response = ollama.embed(
            model='nomic-embed-text',
            input=[question]
        )
        q_embed = np.array(response['embeddings']).astype("float32")
        _, indices = index.search(q_embed, k=2)
        return [chunks[i] for i in indices[0]]
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return []

# 💬 Ask LLM (Local via Ollama)
def ask_llm(context_chunks, question):
    context = "\n\n".join(context_chunks)

    prompt = f"""
    Answer based only on this context:
    {context}

    Question: {question}
    """

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"


# --- App Layout ---

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Top header
st.markdown('<div style="text-align: center;"><h1 class="title-gradient">AI Neural Doc Explorer</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #888; text-align: center; margin-bottom: 2rem;'>Powered by local Llama-3.2 and nomic-embed-text</p></div>", unsafe_allow_html=True)

# 🗂️ Central Document Upload
with st.container():
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="title-gradient" style="margin-top: 0; margin-bottom: 20px;">Data Interface</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Initialize Neural Pattern (PDF)", type="pdf", label_visibility="collapsed")
    
    if uploaded_file:
        if st.button("Initialize Neural Sync"):
            with st.spinner("Extracting parameters..."):
                text = read_pdf(uploaded_file)
                if not text.strip():
                    st.error("Matrix empty. Could not extract data.")
                else:
                    chunks = make_chunks(text)
                    embeddings = get_embeddings(chunks)
                    
                    if len(embeddings) == 0:
                        st.error("Embedding generation failed.")
                    else:
                        index = build_index(embeddings)
                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.success("Neural Sync Complete.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid rgba(0, 243, 255, 0.1); margin: 2rem 0;'>", unsafe_allow_html=True)

# 💬 Main Chat Area
if "index" in st.session_state:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Query the document matrix..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Accessing matrix..."):
                best_chunks = search(
                    prompt,
                    st.session_state.chunks,
                    st.session_state.index
                )
                if best_chunks:
                    response_text = ask_llm(best_chunks, prompt)
                else:
                    response_text = "I couldn't find relevant information in the uploaded document."
                
                st.markdown(response_text)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

else:
    # Empty state UI
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; opacity: 0.5; padding: 2rem;">
            <h2 style="color: #00f3ff; font-size: 2rem; margin-bottom: 0;">Awaiting Matrix Input</h2>
            <p style="font-size: 1.2rem;">Please upload and initialize a PDF above to begin the chat.</p>
        </div>
    """, unsafe_allow_html=True)
    #####python -m streamlit run app.py