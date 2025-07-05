from fpdf import FPDF

class UnicodePDF(FPDF):
    def __init__(self, video_id, object_id, video_filename):
        super().__init__()
        self.video_id = video_id
        self.object_id = object_id
        self.video_filename = video_filename

    def header(self):
        self.set_font("DejaVu", size=12)
        self.cell(0, 10, "Subtitles Extracted from Video", ln=True, align="C")
        self.ln(5)
        self.set_font("DejaVu", size=10)
        self.cell(0, 10, f"Video ID: {self.video_id}", ln=True, align="C")
        self.cell(0, 10, f"Object ID: {self.object_id}", ln=True, align="C")
        self.cell(0, 10, f"Video Name: {self.video_filename}", ln=True, align="C")
        self.ln(5)