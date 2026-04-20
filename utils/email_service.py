"""
utils/email_service.py
────────────────────────────────────────────────────────────────────────────────
CareerLens — Gmail SMTP Email Report Service
────────────────────────────────────────────────────────────────────────────────
Sends a fully formatted HTML email report with:
  - ATS score summary
  - Sub-score breakdown table
  - Top improvement tips
  - Missing keywords
  - Company fit ranking

Setup (one-time):
  1. Enable 2-Factor Authentication on your Gmail account
  2. Go to: myaccount.google.com/apppasswords
  3. Create an App Password for "Mail"
  4. Copy the 16-character password
  5. Create a .env file in the project root:

       GMAIL_SENDER=your.email@gmail.com
       GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
       CAREERLENS_BASE_URL=http://127.0.0.1:8000

  Done. The email service will auto-load these on startup.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GMAIL_SENDER       = os.getenv("GMAIL_SENDER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
BASE_URL           = os.getenv("CAREERLENS_BASE_URL", "http://127.0.0.1:8000")

GRADE_COLORS = {
    "Excellent":        "#2ecc85",
    "Good":             "#4f8ef7",
    "Average":          "#f7c948",
    "Needs Improvement":"#ff4e6a",
}


def _score_color(score: float) -> str:
    if score >= 75: return "#2ecc85"
    if score >= 55: return "#4f8ef7"
    if score >= 40: return "#f7c948"
    return "#ff4e6a"


def build_html_report(analysis: dict, recipient_name: str = "") -> str:
    """Build a beautiful HTML email from the full-analysis API response."""
    ats        = analysis.get("ats_score", {})
    overall    = ats.get("overall", 0)
    grade      = ats.get("grade", "—")
    ss         = ats.get("sub_scores", {})
    tips       = analysis.get("improvement_tips", [])[:5]
    cat        = analysis.get("category", {}).get("predicted_category", "—")
    conf       = analysis.get("category", {}).get("confidence", 0)
    fit        = analysis.get("company_fit_ranking", {})
    parsed     = analysis.get("parsed_info", {})
    grade_color = GRADE_COLORS.get(grade, "#4f8ef7")
    score_color = _score_color(overall)
    now         = datetime.now().strftime("%d %b %Y, %I:%M %p")
    name_line   = f"Hi {recipient_name}," if recipient_name else "Hi there,"

    # Sub-score rows
    sub_rows = ""
    sub_labels = [
        ("🔑 Keyword Match",  "keyword"),
        ("📐 Formatting",     "formatting"),
        ("✅ Completeness",   "completeness"),
        ("📖 Readability",    "readability"),
        ("⚡ Action Verbs",   "action_verbs"),
    ]
    for label, key in sub_labels:
        val = ss.get(key, 0)
        color = _score_color(val)
        bar_width = int(val)
        sub_rows += f"""
        <tr>
          <td style="padding:10px 0;font-size:14px;color:#8888a8;">{label}</td>
          <td style="padding:10px 0;width:200px;">
            <div style="background:#1e1e35;border-radius:4px;height:6px;overflow:hidden;">
              <div style="background:{color};width:{bar_width}%;height:100%;border-radius:4px;"></div>
            </div>
          </td>
          <td style="padding:10px 0;text-align:right;font-weight:700;font-size:15px;color:{color};">{val:.0f}</td>
        </tr>"""

    # Tips rows
    tips_html = ""
    priority_colors = {"High": "#ff4e6a", "Medium": "#f7c948", "Low": "#2ecc85"}
    for tip in tips:
        p = tip.get("priority", "Medium")
        pc = priority_colors.get(p, "#f7c948")
        tips_html += f"""
        <tr>
          <td style="padding:12px 0;border-bottom:1px solid #1e1e35;vertical-align:top;">
            <span style="background:{pc}18;color:{pc};border:1px solid {pc}40;
                         padding:3px 9px;border-radius:5px;font-size:11px;font-weight:600;
                         letter-spacing:.05em;margin-right:10px;">{p.upper()}</span>
            <span style="font-size:13px;color:#8888a8;text-transform:uppercase;
                         letter-spacing:.05em;">{tip.get('category','')}</span>
            <div style="margin-top:6px;font-size:14px;color:#d0d0e8;line-height:1.6;">
              {tip.get('recommendation', tip.get('issue', ''))}
            </div>
            <div style="margin-top:4px;font-size:12px;color:#2ecc85;">
              Expected: {tip.get('expected_improvement','')}
            </div>
          </td>
        </tr>"""

    # Company fit top 5
    fit_html = ""
    for company, data in list(fit.items())[:5]:
        pct = data.get("fit_score", 0)
        c = _score_color(pct)
        fit_html += f"""
        <tr>
          <td style="padding:8px 0;font-size:13px;color:#8888a8;width:110px;">{company}</td>
          <td style="padding:8px 0;">
            <div style="background:#1e1e35;border-radius:3px;height:5px;overflow:hidden;">
              <div style="background:{c};width:{int(pct)}%;height:100%;border-radius:3px;"></div>
            </div>
          </td>
          <td style="padding:8px 0;text-align:right;font-size:13px;font-weight:700;color:{c};width:48px;">{pct:.0f}%</td>
        </tr>"""

    # Missing keywords
    missing_kw = ats.get("missing_keywords", [])[:10]
    kw_pills = "".join(
        f'<span style="display:inline-block;margin:3px;padding:4px 11px;border-radius:14px;'
        f'font-size:12px;font-weight:600;background:rgba(255,78,106,0.1);'
        f'border:1px solid rgba(255,78,106,0.25);color:#ff4e6a;">{k}</span>'
        for k in missing_kw
    ) or '<span style="color:#2ecc85;font-size:13px;">All keywords matched! 🎉</span>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>CareerLens Report</title>
</head>
<body style="margin:0;padding:0;background:#080810;font-family:'Segoe UI',Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#080810;padding:30px 0;">
  <tr><td align="center">
  <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;">

    <!-- HEADER -->
    <tr><td style="background:linear-gradient(135deg,#0f0f1a 0%,#16162a 100%);
                   border:1px solid rgba(255,255,255,0.08);border-radius:16px 16px 0 0;
                   padding:36px 40px;text-align:center;">
      <div style="display:inline-block;width:10px;height:10px;border-radius:50%;
                  background:#ff6b35;box-shadow:0 0 12px #ff6b35;margin-right:8px;
                  vertical-align:middle;"></div>
      <span style="font-size:22px;font-weight:800;color:#eeeef5;letter-spacing:-.5px;vertical-align:middle;">
        Career<span style="color:#ff6b35;">Lens</span>
      </span>
      <div style="margin-top:6px;font-size:12px;color:#8888a8;letter-spacing:.1em;text-transform:uppercase;">
        AI Resume Intelligence Report
      </div>
      <div style="margin-top:4px;font-size:11px;color:#44445a;">{now}</div>
    </td></tr>

    <!-- SCORE HERO -->
    <tr><td style="background:#0f0f1a;border-left:1px solid rgba(255,255,255,0.08);
                   border-right:1px solid rgba(255,255,255,0.08);padding:36px 40px;">
      <p style="color:#8888a8;font-size:14px;margin:0 0 24px;">{name_line}</p>
      <p style="color:#d0d0e8;font-size:14px;line-height:1.7;margin:0 0 28px;">
        Here's your CareerLens resume analysis. Your overall ATS score is:
      </p>

      <div style="text-align:center;padding:32px;background:#16162a;border-radius:14px;
                  border:1px solid rgba(255,255,255,0.07);margin-bottom:28px;">
        <div style="font-size:68px;font-weight:800;line-height:1;color:{score_color};
                    letter-spacing:-.03em;">{overall}</div>
        <div style="font-size:16px;color:#8888a8;margin-top:4px;">out of 100</div>
        <div style="display:inline-block;margin-top:14px;padding:8px 20px;border-radius:8px;
                    background:{grade_color}18;color:{grade_color};
                    border:1px solid {grade_color}45;font-weight:700;font-size:15px;">
          {grade}
        </div>
        <div style="margin-top:12px;font-size:13px;color:#8888a8;">
          Category Detected: <strong style="color:#d0d0e8;">{cat}</strong>
          &nbsp;·&nbsp; Confidence: <strong style="color:{score_color};">{conf:.0f}%</strong>
        </div>
      </div>

      <!-- SUB-SCORES -->
      <h3 style="font-size:14px;font-weight:700;color:#8888a8;text-transform:uppercase;
                 letter-spacing:.08em;margin:0 0 14px;">Sub-Score Breakdown</h3>
      <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
        {sub_rows}
      </table>
    </td></tr>

    <!-- TIPS -->
    <tr><td style="background:#0f0f1a;border-left:1px solid rgba(255,255,255,0.08);
                   border-right:1px solid rgba(255,255,255,0.08);
                   border-top:1px solid rgba(255,255,255,0.06);padding:28px 40px;">
      <h3 style="font-size:14px;font-weight:700;color:#8888a8;text-transform:uppercase;
                 letter-spacing:.08em;margin:0 0 16px;">Top Improvement Tips</h3>
      <table width="100%" cellpadding="0" cellspacing="0">
        {tips_html if tips_html else '<tr><td style="color:#2ecc85;font-size:14px;padding:12px 0;">🎉 No major issues found! Your resume looks great.</td></tr>'}
      </table>
    </td></tr>

    <!-- MISSING KEYWORDS -->
    <tr><td style="background:#0f0f1a;border-left:1px solid rgba(255,255,255,0.08);
                   border-right:1px solid rgba(255,255,255,0.08);
                   border-top:1px solid rgba(255,255,255,0.06);padding:28px 40px;">
      <h3 style="font-size:14px;font-weight:700;color:#8888a8;text-transform:uppercase;
                 letter-spacing:.08em;margin:0 0 14px;">Missing Keywords</h3>
      <div>{kw_pills}</div>
    </td></tr>

    <!-- COMPANY FIT -->
    <tr><td style="background:#0f0f1a;border-left:1px solid rgba(255,255,255,0.08);
                   border-right:1px solid rgba(255,255,255,0.08);
                   border-top:1px solid rgba(255,255,255,0.06);padding:28px 40px;">
      <h3 style="font-size:14px;font-weight:700;color:#8888a8;text-transform:uppercase;
                 letter-spacing:.08em;margin:0 0 14px;">Company Fit (Top 5)</h3>
      <table width="100%" cellpadding="0" cellspacing="0">
        {fit_html if fit_html else '<tr><td style="color:#8888a8;font-size:13px;">No company fit data available.</td></tr>'}
      </table>
    </td></tr>

    <!-- CTA -->
    <tr><td style="background:#0f0f1a;border-left:1px solid rgba(255,255,255,0.08);
                   border-right:1px solid rgba(255,255,255,0.08);
                   border-top:1px solid rgba(255,255,255,0.06);padding:28px 40px;text-align:center;">
      <a href="{BASE_URL}" style="display:inline-block;padding:14px 32px;background:#ff6b35;
                                   border-radius:10px;color:#ffffff;font-weight:700;
                                   font-size:14px;text-decoration:none;letter-spacing:.02em;">
        Open CareerLens Dashboard →
      </a>
    </td></tr>

    <!-- FOOTER -->
    <tr><td style="background:#0a0a14;border:1px solid rgba(255,255,255,0.06);
                   border-radius:0 0 16px 16px;padding:24px 40px;text-align:center;">
      <div style="font-size:12px;color:#44445a;line-height:1.8;">
        CareerLens — AI Resume Intelligence<br/>
        Built by <strong style="color:#8888a8;">Spoorthi S Patil</strong>
        &nbsp;·&nbsp; ML Portfolio Project 2025<br/>
        <span style="color:#2a2a42;">You received this because you requested a report from CareerLens.</span>
      </div>
    </td></tr>

  </table>
  </td></tr>
</table>
</body>
</html>"""

    return html


def send_report(
    recipient_email: str,
    analysis: dict,
    recipient_name: str = "",
) -> dict:
    """
    Send the CareerLens HTML report via Gmail SMTP.

    Returns: {"success": bool, "message": str}
    """
    if not GMAIL_SENDER or not GMAIL_APP_PASSWORD:
        return {
            "success": False,
            "message": (
                "Email not configured. Add GMAIL_SENDER and GMAIL_APP_PASSWORD "
                "to your .env file. See utils/email_service.py for instructions."
            )
        }

    if not recipient_email or "@" not in recipient_email:
        return {"success": False, "message": "Invalid recipient email address."}

    try:
        overall = analysis.get("ats_score", {}).get("overall", 0)
        grade   = analysis.get("ats_score", {}).get("grade", "")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"CareerLens Report — Your ATS Score: {overall}/100 ({grade})"
        msg["From"]    = f"CareerLens <{GMAIL_SENDER}>"
        msg["To"]      = recipient_email

        # Plain text fallback
        plain = (
            f"CareerLens Resume Analysis Report\n"
            f"{'='*40}\n"
            f"Overall ATS Score: {overall}/100 ({grade})\n\n"
            f"View your full report at: {BASE_URL}\n\n"
            f"Built by Spoorthi S Patil — CareerLens AI"
        )
        msg.attach(MIMEText(plain, "plain"))

        # Rich HTML report
        html = build_html_report(analysis, recipient_name)
        msg.attach(MIMEText(html, "html"))

        # Send via Gmail SMTP SSL
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_SENDER, recipient_email, msg.as_string())

        return {
            "success": True,
            "message": f"Report successfully sent to {recipient_email}"
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "message": (
                "Gmail authentication failed. Make sure you're using an App Password, "
                "not your regular Gmail password. See setup instructions in email_service.py."
            )
        }
    except smtplib.SMTPException as e:
        return {"success": False, "message": f"SMTP error: {str(e)}"}
    except Exception as e:
        return {"success": False, "message": f"Failed to send email: {str(e)}"}


def is_configured() -> bool:
    """Check if email credentials are set up."""
    return bool(GMAIL_SENDER and GMAIL_APP_PASSWORD)


if __name__ == "__main__":
    # Quick config check
    print(f"Email configured: {is_configured()}")
    print(f"Sender:           {GMAIL_SENDER or '(not set)'}")
    print(f"App password:     {'✓ set' if GMAIL_APP_PASSWORD else '(not set)'}")

    if not is_configured():
        print("\nTo configure:")
        print("  1. Enable 2FA on Gmail")
        print("  2. Go to myaccount.google.com/apppasswords")
        print("  3. Create App Password → Mail")
        print("  4. Create .env file:")
        print("       GMAIL_SENDER=your@gmail.com")
        print("       GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx")
    else:
        print("\nReady to send emails! Test with:")
        print("  python -c \"from utils.email_service import send_report; print(send_report('test@example.com', {}))\"")
