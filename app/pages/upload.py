from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import tempfile
import uuid
import zipfile

import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auth_ui import configure_page, get_auth_user_id, get_robot_image
from database import (
    create_session_record,
    get_max_session_id,
    get_storage_paths,
    init_db,
    update_session_status,
    upsert_storage_paths,
)
from PIPELINE.preprocessing import (
    validate_and_prepare_audio,
    validate_portrait_image,
    validate_source_document,
)
from PIPELINE.session_manager import create_session, get_next_session_number


def _upload_css() -> None:
    # Upload page-specific styling.
    # This CSS controls the title look, the robot placement, the right-side
    # container widths, the section rectangle borders, and button styling.
    st.markdown(
        """
        <style>
        /* Controls hiding the default Streamlit header/sidebar/footer. */
        .stApp > header, header[data-testid="stHeader"], footer,
        [data-testid="stSidebar"], [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        /* Controls the base font size across the whole page. */
        html {
            font-size: 70px !important;
        }
        /* Controls the page background color. */
        html, body, .stApp {
            background: #ffffff !important;
        }
        /* Controls the outer page padding and max page width. */
        [data-testid="stAppViewBlockContainer"], .block-container {
            padding-top: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;
        }
        /* Controls the title block position. */
        .upload-title {
            margin: -20px 0 0 15px;
        }
        /* Controls the TALEXA title font, size, and color. */
        .upload-title h1 {
            color: #1b1f8f;
            font-family: Arial Black, Arial, sans-serif;
            font-size: 300px;
            margin: 0;
            line-height: 1;
        }
        /* Controls the 4 main rectangle containers on the right. */
        .st-key-upload_text_section,
        .st-key-upload_portrait_section,
        .st-key-upload_audio_section,
        .st-key-upload_language_section {
            border: 10px solid #1b1f8f;
            border-radius: 6px;
            background: #ffffff;
            padding: 0 20px;
            margin-top: -40px;
            margin-bottom: 10px;
            margin-left: auto;
            margin-right: 600px;
            position: relative;
            width: 70% !important;
            max-width: 70% !important;
            min-height: 220px;
            text-align: center;
        }
        .st-key-upload_text_section,
        .st-key-upload_portrait_section,
        .st-key-upload_audio_section,
        .st-key-upload_language_section,
        .st-key-upload_text_section *,
        .st-key-upload_portrait_section *,
        .st-key-upload_audio_section *,
        .st-key-upload_language_section * {
            font-family: "Times New Roman", Times, serif !important;
            font-weight: 700 !important;
            text-align: center !important;
        }
        .st-key-upload_text_section [data-testid="stMarkdownContainer"],
        .st-key-upload_portrait_section [data-testid="stMarkdownContainer"],
        .st-key-upload_audio_section [data-testid="stMarkdownContainer"],
        .st-key-upload_language_section [data-testid="stMarkdownContainer"] {
            text-align: center !important;
            width: 100% !important;
        }
        .st-key-upload_text_section [data-testid="stMarkdownContainer"] p,
        .st-key-upload_portrait_section [data-testid="stMarkdownContainer"] p,
        .st-key-upload_audio_section [data-testid="stMarkdownContainer"] p,
        .st-key-upload_language_section [data-testid="stMarkdownContainer"] p {
            text-align: center !important;
            width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .st-key-text_pdf,
        .st-key-portrait_png,
        .st-key-audio_wav,
        .st-key-text_pdf[data-testid="stElementContainer"],
        .st-key-portrait_png[data-testid="stElementContainer"],
        .st-key-audio_wav[data-testid="stElementContainer"] {
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-text_pdf [data-testid="stFileUploader"],
        .st-key-portrait_png [data-testid="stFileUploader"],
        .st-key-audio_wav [data-testid="stFileUploader"] {
            position: relative !important;
            width: 100% !important;
            min-height: 60px !important;
            margin: 0 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        .st-key-text_pdf [data-testid="stFileUploaderDropzone"],
        .st-key-portrait_png [data-testid="stFileUploaderDropzone"],
        .st-key-audio_wav [data-testid="stFileUploaderDropzone"] {
            position: absolute !important;
            inset: 0 !important;
            border: none !important;
            background: transparent !important;
            padding: 0 !important;
            min-height: 60px !important;
            width: 100% !important;
        }
        .st-key-text_pdf [data-testid="stFileUploaderDropzone"] > div,
        .st-key-portrait_png [data-testid="stFileUploaderDropzone"] > div,
        .st-key-audio_wav [data-testid="stFileUploaderDropzone"] > div {
            width: 100% !important;
            height: 60px !important;
        }
        .st-key-upload_text_section .stRadio,
        .st-key-upload_language_section .stRadio {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            width: 100% !important;
        }
        .st-key-upload_text_section .stRadio > label,
        .st-key-upload_language_section .stRadio > label {
            width: 100% !important;
            text-align: center !important;
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .st-key-upload_text_section .stRadio [role="radiogroup"],
        .st-key-upload_language_section .stRadio [role="radiogroup"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
        }
        .st-key-upload_text_section > div,
        .st-key-upload_portrait_section > div,
        .st-key-upload_audio_section > div,
        .st-key-upload_language_section > div {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            width: 100% !important;
            padding-top: 14px !important;
        }
        /* Controls horizontal layout for any title/button row inside a container. */
        .upload-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 8px;
        }
        /* Controls smaller row-title text styling. */
        .upload-row-title {
            color: #16206f;
            font-family: "Times New Roman", Times, serif;
            font-size: 80px;
            font-weight: 700;
            margin: 0;
        }
        /* Controls the main title text inside each right-side container. */
        .upload-card-title {
            color: #16206f;
            font-family: "Times New Roman", Times, serif;
            font-size: 78px;
            font-weight: 700;
            line-height: 1.1;
            margin: 50px 0 0 !important;
        }
        .upload-title-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            width: 100%;
            margin-bottom: 6px;
        }
        /* Controls the helper note text under each container title. */
        .section-note {
            color: #2d357f;
            font-family: "Times New Roman", Times, serif;
            font-size: 60px;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        /* Controls the current-session badge above the form. */
        .session-chip {
            display: inline-block;
            background: #18208f;
            color: white;
            border-radius: 999px;
            padding: 8px 16px;
            font-family: "Times New Roman", Times, serif;
            font-size: 58px;
            margin-bottom: 16px;
        }
        /* Controls the whole upload form block position and width. */
        .right-panel {
            padding-top: 70;
            margin-top: -220px;
            max-width: 12px;
            width: 100%;
        }
        /* Controls the robot block vertical position. */
        .robot-panel {
            padding-top: 480px;
        }
        /* Controls the robot image left/center/right alignment. */
        .robot-panel [data-testid="stImage"] {
            display: flex;
            justify-content: flex-start;
        }
        /* Controls the robot image container width. */
        .robot-panel [data-testid="stImageContainer"] {
            width: 100% !important;
            max-width: 460px !important;
        }
        /* Controls the actual robot image size. */
        .robot-panel [data-testid="stImageContainer"] img {
            width: 100% !important;
            max-width: 460px !important;
            height: auto !important;
            object-fit: contain !important;
        }
        /* Controls spacing between download buttons. */
        .download-spacer {
            height: 12px;
        }
        /* Controls spacing around uploader widgets. */
        [data-testid="stFileUploader"] {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        /* Removes the default dashed uploader box. */
        [data-testid="stFileUploaderDropzone"] {
            border: none !important;
            background: transparent !important;
            padding: 0 !important;
            min-height: auto !important;
        }
        /* Controls the visible upload button style. */
        [data-testid="stFileUploaderDropzone"] button {
            border: none !important;
            background: transparent !important;
            color: transparent !important;
            border-radius: 0 !important;
            min-height: 60px !important;
            width: 100% !important;
            padding: 0 !important;
            font-size: 0 !important;
            line-height: 1 !important;
            font-weight: 700 !important;
            opacity: 0 !important;
            cursor: pointer !important;
        }
        [data-testid="stFileUploaderDropzone"] button *,
        [data-testid="stFileUploaderDropzone"] button span {
            font-size: 0 !important;
            line-height: 1 !important;
            color: transparent !important;
        }
        /* Hides Streamlit's default uploader instructions/labels. */
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] div:first-child,
        [data-testid="stFileUploader"] > label {
            display: none !important;
        }
        /* Re-show the three uploader labels and style them like clickable titles. */
        .st-key-text_pdf [data-testid="stFileUploader"] > label,
        .st-key-portrait_png [data-testid="stFileUploader"] > label,
        .st-key-audio_wav [data-testid="stFileUploader"] > label {
            display: block !important;
            text-align: center !important;
            color: #16206f !important;
            font-family: "Times New Roman", Times, serif !important;
            font-size: 78px !important;
            font-weight: 700 !important;
            line-height: 1.1 !important;
            margin: 50px auto 0 !important;
            min-height: 60px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            cursor: pointer !important;
        }
        .st-key-text_pdf [data-testid="stFileUploader"] > label *,
        .st-key-portrait_png [data-testid="stFileUploader"] > label *,
        .st-key-audio_wav [data-testid="stFileUploader"] > label * {
            font-family: "Times New Roman", Times, serif !important;
            font-size: 78px !important;
            font-weight: 700 !important;
            line-height: 1.1 !important;
        }
        /* Controls the radio group label style. */
        .stRadio > label {
            font-size: 54px !important;
            color: #1b1f8f !important;
            font-weight: 700 !important;
            font-family: "Times New Roman", Times, serif !important;
            margin-bottom: 0 !important;
        }
        .st-key-upload_language_section .stRadio > label {
            display: none !important;
        }
        /* Controls the individual radio option text style. */
        .stRadio [role="radiogroup"] label {
            font-size: 55px !important;
            color: #1d255f !important;
            font-family: "Times New Roman", Times, serif !important;
            font-weight: 700 !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 8px !important;
            margin: 0 !important;
        }
        .stRadio [role="radiogroup"] label p,
        .stRadio [role="radiogroup"] label span,
        .stRadio [role="radiogroup"] label div {
            font-size: 55px !important;
            font-family: "Times New Roman", Times, serif !important;
            font-weight: 700 !important;
            line-height: 1.1 !important;
            margin: 0 !important;
        }
        .stRadio [role="radiogroup"] {
            gap: 6px !important;
            margin-top: 0 !important;
        }
        .stRadio [role="radiogroup"] input[type="radio"] {
            width: 18px !important;
            height: 18px !important;
            margin: 0 !important;
        }
        /* Controls the Generate button and download button styling. */
        div.stButton > button,
        div[data-testid="stDownloadButton"] > button {
            border: 10px solid #1b1f8f !important;
            background: transparent !important;
            color: #111111 !important;
            border-radius: 8px !important;
            min-height: 56px !important;
            font-family: "Times New Roman", Times, serif !important;
            font-size: 58px !important;
            font-weight: 700 !important;
        }
        div.stButton > button *,
        div[data-testid="stDownloadButton"] > button * {
            font-family: "Times New Roman", Times, serif !important;
            font-weight: 700 !important;
        }
        /* Controls button hover colors. */
        div.stButton > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {
            background: #1b1f8f !important;
            color: #ffffff !important;
        }
        /* Controls the spacing around the Generate button row. */
        .generate-row {
            margin: 18px 0 22px;
        }
        .generate-row [data-testid="stButton"] {
            margin-left: -20px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _save_uploaded_file(uploaded_file, suffix: str) -> Path:
    # Save each uploaded file to a temporary staging path before validation/copying.
    staging_dir = Path(tempfile.gettempdir()) / "talexa_upload_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    target_path = staging_dir / f"{uuid.uuid4().hex}_{safe_name}"
    if suffix and target_path.suffix.lower() != suffix.lower():
        target_path = target_path.with_suffix(suffix)
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def _zip_directory(directory: Path) -> bytes:
    # Package generated video outputs into a single downloadable zip.
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(directory))
    memory_file.seek(0)
    return memory_file.getvalue()


def _existing_output_paths(session_id: int | None) -> tuple[Path | None, Path | None]:
    # Resolve any stored output paths for the current/generated session.
    if not session_id:
        return None, None

    record = get_storage_paths(session_id)
    if record is None:
        return None, None

    slides_path = Path(record["slides_output_path"]) if record["slides_output_path"] else None
    video_path = Path(record["video_output_path"]) if record["video_output_path"] else None
    return slides_path, video_path


configure_page("TALEXA Upload")
init_db()

# Apply the upload page styling before rendering any UI elements.
_upload_css()

# Render the page title at the top-left of the upload screen.
st.markdown(
    """
    <div class="upload-title">
        <h1>TALEXA</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Read the latest generated/restored session id from the URL, if present.
latest_session_id = st.query_params.get("session_id")
# Keep transient status feedback in local variables for this render cycle.
upload_status_message: str | None = None
upload_status_type = "info"

# Show the current session chip when a session id exists in the URL.
if latest_session_id:
    st.markdown(
        f'<div class="session-chip">Current Session: {latest_session_id}</div>',
        unsafe_allow_html=True,
    )

# Split the page into the robot area on the left and the upload controls on the right.
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # Left side: robot image.
    # This wrapper uses the `.robot-panel` CSS block above.
    st.markdown('<div class="robot-panel">', unsafe_allow_html=True)
    robot_image = get_robot_image()
    if robot_image and robot_image.exists():
        # This is the actual robot image element.
        # Its size is controlled by the robot image CSS selectors above.
        st.image(str(robot_image), use_container_width=True)
    else:
        st.info("Robot image not found in the assets folder.")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    # Right side: upload controls and actions.
    # This wrapper uses the `.right-panel` CSS block above.
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    # --- TEXT CONTAINER ---
    # Choose the source type and upload a single PDF.
    # This block collects the lesson source and decides whether the slides
    # pipeline or textbook pipeline will run after Generate is clicked.
    # The bordered rectangle for this block is created by the keyed container below.
    with st.container(key="upload_text_section"):
        # The uploader label is the visible clickable "Upload Text" title.
        text_file = st.file_uploader(
            "Upload Text",
            type=["pdf"],
            accept_multiple_files=False,
            key="text_pdf",
        )
        # These are the text type options.
        # This control chooses between the slides pipeline and textbook pipeline.
        text_type = st.radio(
            "Text type",
            options=["slides", "textbook"],
            format_func=lambda option: "Slides" if option == "slides" else "Textbook Chapter",
            horizontal=True,
        )

    # --- PORTRAIT CONTAINER ---
    # Upload one portrait image in PNG format.
    # This image is validated and stored inside the session input folder.
    # The bordered rectangle for this block is created by the keyed container below.
    with st.container(key="upload_portrait_section"):
        # The uploader label is the visible clickable "Upload Portrait" title.
        portrait_file = st.file_uploader(
            "Upload Portrait",
            type=["png"],
            accept_multiple_files=False,
            key="portrait_png",
        )
    # --- AUDIO CONTAINER ---
    # Upload one WAV audio sample.
    # This audio is validated, normalized by preprocessing, and saved to the session.
    # The bordered rectangle for this block is created by the keyed container below.
    with st.container(key="upload_audio_section"):
        # The uploader label is the visible clickable "Upload Audio Sample" title.
        audio_file = st.file_uploader(
            "Upload Audio Sample",
            type=["wav"],
            accept_multiple_files=False,
            key="audio_wav",
        )
    # --- LANGUAGE CONTAINER ---
    # Choose the lecture output language.
    # The selected language is passed into the pipeline and stored in the session record.
    # The bordered rectangle for this block is created by the keyed container below.
    with st.container(key="upload_language_section"):
        # This is the visible title text for container 4.
        st.markdown('<div class="upload-card-title">Choose Language</div>', unsafe_allow_html=True)
        # These are the language options for the final generated output.
        language = st.radio(
            "Lecture language",
            options=["english", "arabic"],
            format_func=lambda option: option.title(),
            horizontal=True,
        )

    # --- GENERATE BUTTON ---
    # Start session creation and run the selected pipeline.
    # When clicked, the app creates the DB session record, saves the uploaded
    # files, runs the pipeline, and stores any output paths for downloads.
    # This row controls where the Generate button sits under the four containers.
    st.markdown('<div class="generate-row">', unsafe_allow_html=True)
    generate_left, generate_mid, generate_right = st.columns([1.0, 1.0, 1.2])
    with generate_mid:
        # This is the main Generate button.
        generate_clicked = st.button("GENERATE", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if generate_clicked:
        # Require all mandatory files before starting session creation.
        if text_file is None or portrait_file is None or audio_file is None:
            upload_status_message = "Please upload the text PDF, portrait PNG, and audio WAV before generating."
            upload_status_type = "error"
        else:
            # Keep the filesystem session folder id and DB session id aligned.
            session_id = max(get_next_session_number(PROJECT_ROOT), get_max_session_id() + 1)
            session_record_created = False

            try:
                # Stage raw uploads in temp storage before validation.
                source_upload_path = _save_uploaded_file(text_file, ".pdf")
                portrait_upload_path = _save_uploaded_file(portrait_file, ".png")
                audio_upload_path = _save_uploaded_file(audio_file, ".wav")

                # Validate and prepare each input using the existing preprocessing layer.
                validated_source_path = validate_source_document(str(source_upload_path), text_type)
                validated_portrait_path = validate_portrait_image(str(portrait_upload_path))
                validated_audio_path = validate_and_prepare_audio(str(audio_upload_path))

                # Create the session record first so the database matches the session folder.
                create_session_record(
                    session_id=session_id,
                    user_id=int(get_auth_user_id()),
                    text_type=text_type,
                    language=language,
                    status="created",
                )
                session_record_created = True

                # Create the session folder structure and copy validated inputs into it.
                session = create_session(
                    PROJECT_ROOT,
                    pdf_file_path=str(validated_source_path),
                    audio_file_path=str(validated_audio_path),
                    portrait_file_path=str(validated_portrait_path),
                    session_number=session_id,
                )

                # Save input file paths in the storage paths table.
                upsert_storage_paths(
                    user_id=int(get_auth_user_id()),
                    session_id=session_id,
                    image_path=str(session.stored_portrait_path) if session.stored_portrait_path else None,
                    audio_path=str(session.stored_audio_path) if session.stored_audio_path else None,
                    slides_path=str(session.stored_pdf_path) if text_type == "slides" else None,
                    textbook_path=str(session.stored_pdf_path) if text_type == "textbook" else None,
                )
                update_session_status(session_id, "running")

                # Import lazily so the upload page can still open even if some
                # pipeline-only dependencies are not installed until Generate is used.
                with st.spinner("Generating slides and lecture assets..."):
                    from PIPELINE.run_pipeline import run_pipeline_for_session

                    # Run the correct pipeline branch based on the selected text type.
                    result = run_pipeline_for_session(
                        session=session,
                        source_type=text_type,
                        lecture_language=language,
                    )

                # Save output paths back into storage once generation completes.
                slides_output_path = result.get("slides_pdf_path")
                video_output_path = result.get("video_output_path")
                upsert_storage_paths(
                    user_id=int(get_auth_user_id()),
                    session_id=session_id,
                    image_path=str(session.stored_portrait_path) if session.stored_portrait_path else None,
                    audio_path=str(session.stored_audio_path) if session.stored_audio_path else None,
                    slides_path=str(session.stored_pdf_path) if text_type == "slides" else None,
                    textbook_path=str(session.stored_pdf_path) if text_type == "textbook" else None,
                    slides_output_path=str(slides_output_path) if slides_output_path else None,
                    video_output_path=str(video_output_path) if video_output_path else None,
                )
                update_session_status(session_id, "completed")

                # Store the latest session id in the URL so downloads can be reloaded.
                st.query_params["session_id"] = str(session_id)
                latest_session_id = str(session_id)
                upload_status_message = f"Session {session_id} was created and the pipeline completed successfully."
                upload_status_type = "success"
            except Exception as exc:
                # Mark the session as failed if creation already reached the database layer.
                if session_record_created:
                    update_session_status(session_id, "failed")
                st.query_params["session_id"] = str(session_id)
                latest_session_id = str(session_id)
                upload_status_message = f"Generation failed: {exc}"
                upload_status_type = "error"

    # --- STATUS AND DOWNLOADS ---
    # Show generation feedback and any available outputs for download.
    # Download buttons only appear when output paths exist and point to real files/folders.
    if upload_status_message:
        if upload_status_type == "success":
            st.success(upload_status_message)
        else:
            st.error(upload_status_message)

    # Resolve any stored outputs for the currently selected session id.
    current_session_id = int(latest_session_id) if latest_session_id else None
    slides_output, video_output = _existing_output_paths(current_session_id)

    # Package the lecture output folder as a zip when video assets exist.
    if video_output and video_output.exists() and video_output.is_dir() and any(video_output.iterdir()):
        lecture_zip = _zip_directory(video_output)
        st.download_button(
            "Download Lecture Here",
            data=lecture_zip,
            file_name=f"session_{current_session_id}_lecture.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.markdown('<div class="download-spacer"></div>', unsafe_allow_html=True)

    # Offer the generated slides PDF directly when it exists.
    if slides_output and slides_output.exists() and slides_output.is_file():
        st.download_button(
            "Download Slides",
            data=slides_output.read_bytes(),
            file_name=slides_output.name,
            mime="application/pdf",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
