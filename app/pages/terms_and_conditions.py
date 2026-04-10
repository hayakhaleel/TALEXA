from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from auth_ui import configure_page, local_css
from database import init_db


configure_page("TALEXA Terms and Conditions")
init_db()

terms_message: str | None = None

local_css()

st.markdown(
    """
    <div style="margin:0 0 20px 20px;">
        <h1 style="
            color:#1b1f8f;
            font-family: Arial Black, Arial, sans-serif;
            font-size:300px;
            margin:0;
            line-height:1;
        ">TALEXA</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div style="height: 400px;"></div>', unsafe_allow_html=True)
form_left, form_mid, form_right = st.columns([2.2, 4.6, 2.2])
with form_mid:
    st.markdown('<div class="terms-title">TERMS AND CONDITIONS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="terms-list">
            <li>By using our platform, you agree that any images, portraits, textbooks, and audio files you upload will be processed solely for the purpose of generating AI-generated slides and lectures.</li>
            <li>All uploaded files are used by the system to process your content and generate the requested lecture.</li>
            <li>Your uploaded content is never shared with third parties.</li>
            <li>By uploading any material, you confirm that you have the legal right to use such content and that it does not violate copyright, intellectual property, or privacy laws.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="terms-button-wrap"></div>', unsafe_allow_html=True)
    if st.button("ACCEPT TERMS AND CONDITIONS", use_container_width=True):
        terms_message = "Terms accepted."
        st.switch_page("pages/upload.py")

    if terms_message:
        st.success(terms_message)

    if st.button("BACK TO LOGIN", use_container_width=True):
        st.switch_page("pages/login.py")
