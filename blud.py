import streamlit as st

def main():
    st.title("Image Printer")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Print the image
        st.write("Printing the uploaded image...")

if __name__ == "__main__":
    main()
