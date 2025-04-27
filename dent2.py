# import streamlit as st

# # Custom CSS for dialog box container
# st.markdown("""
#     <style>
#         .dialog-box {
#             background-color: #f0f2f6;
#             border: 2px solid #d0d7de;
#             border-radius: 15px;
#             padding: 20px;
#             max-width: 500px;
#             margin: 20px auto;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#             font-family: Arial, sans-serif;
#             font-size: 16px;
#             color: #333;
#         }

#         .dialog-box h4 {
#             margin-top: 0;
#             color: #004085;
#         }

#         .dialog-box p {
#             margin: 0;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # HTML content for dialog box
# dialog_html = """
# <div class="dialog-box">
#     <h4>System Message</h4>
#     <p>This is a custom dialog box styled with CSS inside Streamlit.</p>
# </div>
# """

# # Render the dialog box
# st.markdown(dialog_html, unsafe_allow_html=True)