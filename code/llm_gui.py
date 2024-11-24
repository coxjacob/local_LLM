import streamlit as st

import a_llm_runner as alr


# Create header
header = st.container()
body = st.container()

with header:
    st.title("Jacob's Local LLM")
    st.write("Protected Querry")

with body:
    st.write("Enter q or Q to quit.")
    entered_text = st.text_input("Enter Querry",'') 

    if entered_text:
        answer = alr.generate_m7b_output(entered_text)
        list_answer = answer.split('\n')
        st.write(list_answer[-1])

