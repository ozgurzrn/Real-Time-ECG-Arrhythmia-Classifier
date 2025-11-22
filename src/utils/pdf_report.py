from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
import matplotlib.pyplot as plt
import io

def generate_pdf_report(results, clean_signal, display_results, display_signal, filename="ecg_report.pdf"):
    """
    Generate a professional PDF report for ECG analysis.
    """
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
    )
    story.append(Paragraph("ECG Arrhythmia Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Beats Analyzed:</b> {len(results)}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary Statistics
    story.append(Paragraph("<b>Detection Summary</b>", styles['Heading2']))
    
    from collections import Counter
    class_counts = Counter([r['pred_idx'] for r in results])
    total = len(results)
    
    class_names = {0: 'Normal (N)', 1: 'Supraventricular (S)', 2: 'Ventricular (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}
    
    # Create summary table
    data = [['Beat Type', 'Count', 'Percentage']]
    for idx, name in class_names.items():
        count = class_counts.get(idx, 0)
        percentage = f"{count/total*100:.1f}%"
        data.append([name, str(count), percentage])
    
    table = Table(data, colWidths=[2.5*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Clinical Assessment
    arrhythmia_count = class_counts.get(1, 0) + class_counts.get(2, 0) + class_counts.get(3, 0)
    arrhythmia_percentage = (arrhythmia_count / total) * 100
    
    story.append(Paragraph("<b>Clinical Assessment</b>", styles['Heading2']))
    
    if arrhythmia_percentage > 5:
        assessment = f"<font color='red'><b>ARRHYTHMIA DETECTED:</b></font> {arrhythmia_count} abnormal beats ({arrhythmia_percentage:.1f}% of total). Recommend further clinical evaluation."
    elif arrhythmia_percentage > 0:
        assessment = f"<font color='orange'><b>Minor Arrhythmia:</b></font> {arrhythmia_count} abnormal beats ({arrhythmia_percentage:.1f}% of total). Monitor patient."
    else:
        assessment = "<font color='green'><b>NORMAL RHYTHM:</b></font> No significant arrhythmias detected."
    
    story.append(Paragraph(assessment, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # ECG Plot
    story.append(Paragraph("<b>ECG Signal (First 10 seconds)</b>", styles['Heading2']))
    
    # Generate plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(display_signal, color='black', linewidth=0.5)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    ax.set_title('ECG Signal')
    ax.grid(True, alpha=0.3)
    
    # Save plot to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    # Add to PDF
    img = Image(img_buffer, width=6*inch, height=2.25*inch)
    story.append(img)
    story.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    story.append(Paragraph("<b>Disclaimer</b>", styles['Heading3']))
    story.append(Paragraph(
        "This automated analysis is for screening purposes only and should not replace professional medical diagnosis. "
        "All findings should be verified by a qualified healthcare provider.",
        styles['Normal']
    ))
    
    # Build PDF
    doc.build(story)
    return filename
