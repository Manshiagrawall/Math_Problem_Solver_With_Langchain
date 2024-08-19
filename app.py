import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
import json

# Set up the Streamlit app
st.set_page_config(page_title="Math Problem Solver & Data Search Assistant", page_icon="🧮", layout="wide")
st.title("🧮 Math Problem Solver and Knowledge Assistant with LangChain")

# Sidebar setup
st.sidebar.title("Configuration")
groq_api_key = st.sidebar.text_input(label="🔑 Groq API Key", type="password")

if not groq_api_key:
    st.sidebar.warning("Please add your Groq API key to continue.")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find various information on the topics mentioned"
)

# Initialize the Math tool with enhanced error handling
def safe_math_chain(expression):
    try:
        return LLMMathChain.from_llm(llm=llm).run(expression)
    except ValueError as e:
        return f"Error: {str(e)}. Please try again with a valid numerical expression."

calculator = Tool(
    name="Calculator",
    func=safe_math_chain,
    description="A tool for answering math-related questions. Only input valid mathematical expressions."
)

prompt = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation,
displaying it point-wise for the question below:
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into a chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Initialize the agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your math questions!"}
    ]

if "question_history" not in st.session_state:
    st.session_state["question_history"] = []

# Sidebar: Previous Questions & Save Conversation
st.sidebar.subheader("📜 Previous Questions")
for idx, q in enumerate(st.session_state["question_history"]):
    if st.sidebar.button(f"Q{idx+1}: {q}"):
        st.session_state.messages.append({"role": "user", "content": q})
        st.chat_message("user").write(q)

# Save conversation to a file
def save_conversation():
    conversation = json.dumps(st.session_state["messages"])
    st.sidebar.download_button("💾 Download Conversation", conversation, file_name="conversation.json", mime="application/json")

save_conversation()

# Display chat history
st.markdown("---")
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Interaction section
st.markdown("### Ask a Question")
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?",
    height=100
)

# Enhanced UI elements for interaction
st.markdown("<style> .stButton > button { background-color: #4CAF50; color: white; } </style>", unsafe_allow_html=True)

if st.button("Find My Answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.session_state["question_history"].append(question)

            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please enter a question.")

# Clear chat button in the sidebar
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your math questions!"}
    ]
    st.session_state["question_history"] = []

# Footer for better visual appeal
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px; color: #888;'>
        Developed by [Your Name]. Powered by LangChain and Google Gemma 2.
    </div>
    """,
    unsafe_allow_html=True
)
