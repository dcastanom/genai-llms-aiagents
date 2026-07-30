import streamlit as st
from main import crew

st.set_page_config(page_title="AI Research and Writing Crew", layout="centered")
st.title("AI Research and Writing Crew")
st.write("Give me a topic, and my AI crew will research and write a blog post for you!")

# use a session state to store the topic and results
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "results" not in st.session_state:
    st.session_state.results = ""

topic = st.text_input("Enter a topic for the blog post:", value=st.session_state.topic, placeholder="e.g., Artificial Intelligence in Healthcare")

if st.button("Generate Blog Post"):
    if topic:
        st.session_state.topic = topic
        with st.spinner("Generating blog post..."):
            results = crew.kickoff(inputs={"topic": topic})
            if hasattr(results, "output"):
                st.session_state.results = results.output
            elif isinstance(results, dict) and "output" in results:
                st.session_state.results = results["output"]
            else:
                st.session_state.results = str(results)
    else:
        st.warning("Please enter a topic before generating the blog post.")

# Displaying the results
if st.session_state.results:
    st.markdown("---")
    st.subheader("Generated Blog Post")
    st.markdown(st.session_state.results)