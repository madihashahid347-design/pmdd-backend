import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import uuid
from schemas.agent_schemas import Agent5Output

class AcademicReportPDF(FPDF):
    def header(self):
        self.set_font("Times", "B", 12)
        self.cell(0, 10, "Pragmatic Meaning Drift Detector (PMDD)", 0, 1, "R")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def generate_chart(scores, output_path):
    """Generates a static bar chart for the Meaning Drift Scores."""
    labels = ['Overall', 'Pragmatic', 'Semantic', 'Register']
    values = [scores.overall, scores.pragmatic, scores.semantic, scores.register]
    
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['#475569', '#3b82f6', '#10b981', '#8b5cf6'])
    plt.ylim(0, 1.0)
    plt.title("Meaning Drift Scores")
    plt.ylabel("Drift Magnitude (0-1)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_scientific_pdf(report_data: Agent5Output, analysis_id: str) -> str:
    """Generates a publication-ready PDF from Agent 5 JSON output."""
    pdf = AcademicReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Times", "B", 18)
    pdf.cell(0, 15, "Scientific Report: Multi-Agent Linguistic Analysis", 0, 1, "C")
    pdf.ln(10)
    
    # Method to render a section
    def add_section(title, content):
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, title.upper(), 0, 1, "L")
        pdf.set_font("Times", "", 11)
        pdf.multi_cell(0, 6, content)
        pdf.ln(5)

    add_section("1. Executive Summary", report_data.executive_summary)
    add_section("2. Corpus Profile", report_data.corpus_profile)
    add_section("3. Pragmatic Drift Evidence", report_data.pragmatic_drift_evidence)
    add_section("4. Semantic Field & Register Analysis", report_data.semantic_register_analysis)
    add_section("5. Quantitative Corpus Statistics", report_data.quantitative_corpus_statistics)
    
    # Generate and embed chart
    chart_filename = f"temp_chart_{uuid.uuid4().hex}.png"
    generate_chart(report_data.scores, chart_filename)
    pdf.image(chart_filename, x=30, w=150)
    os.remove(chart_filename) # Cleanup
    pdf.ln(10)
    
    add_section("6. Multi-Agent Synthesis", report_data.multi_agent_synthesis)
    add_section("7. Linguistic Theory Mapping", report_data.linguistic_theory_mapping)
    add_section("8. Meaning Drift Score & Conclusions", report_data.conclusions)
    add_section("9. Methodology", report_data.methodology)
    add_section("10. References", report_data.references)
    add_section("11. Appendix", report_data.appendix)
    
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{analysis_id}.pdf")
    pdf.output(pdf_path)
    
    return pdf_path
