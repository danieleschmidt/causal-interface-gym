#!/usr/bin/env python3
"""
Generate comprehensive project reports for Causal Interface Gym.

This script generates various types of reports based on collected metrics
and project data.

Usage:
    python scripts/generate_report.py --type health
    python scripts/generate_report.py --type security --format pdf
    python scripts/generate_report.py --type performance --period weekly
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class ReportGenerator:
    """Generates various project reports."""
    
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(__file__).parent.parent
        self.metrics_file = self.base_path / ".github" / "project-metrics.json"
        self.output_dir = self.base_path / "reports"
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load project metrics."""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                return json.load(f)
        return {}
    
    def generate_health_report(self, format_type: str = "markdown") -> str:
        """Generate project health report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if format_type == "markdown":
            return self._generate_health_markdown(timestamp)
        elif format_type == "html":
            return self._generate_health_html(timestamp)
        elif format_type == "pdf":
            return self._generate_health_pdf(timestamp)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_health_markdown(self, timestamp: str) -> str:
        """Generate health report in Markdown format."""
        health = self.metrics.get("health", {})
        overall_score = health.get("overall_score", 0)
        categories = health.get("categories", {})
        
        # Determine health status
        if overall_score >= 90:
            status = "🟢 Excellent"
            status_emoji = "🎉"
        elif overall_score >= 80:
            status = "🟡 Good"
            status_emoji = "👍"
        elif overall_score >= 70:
            status = "🟠 Fair"
            status_emoji = "⚠️"
        else:
            status = "🔴 Needs Attention"
            status_emoji = "🚨"
        
        report = f"""# Causal Interface Gym - Health Report

*Generated: {timestamp}*

## {status_emoji} Overall Health: {overall_score:.1f}/100 - {status}

### Health Categories

| Category | Score | Status |
|----------|-------|--------|
"""
        
        for category, score in categories.items():
            if score >= 80:
                cat_status = "✅ Good"
            elif score >= 60:
                cat_status = "⚠️ Fair"
            else:
                cat_status = "❌ Poor"
            
            category_name = category.replace("_", " ").title()
            report += f"| {category_name} | {score:.1f} | {cat_status} |\n"
        
        # Add detailed sections
        report += "\n## 📊 Detailed Metrics\n\n"
        
        # Code Quality
        code_metrics = self.metrics.get("metrics", {}).get("code", {})
        if code_metrics:
            report += "### Code Quality\n"
            loc = code_metrics.get("lines_of_code", {})
            coverage = code_metrics.get("test_coverage", {})
            
            report += f"- **Lines of Code**: {loc.get('total', 0):,}\n"
            report += f"  - Python: {loc.get('python', 0):,}\n"
            report += f"  - TypeScript: {loc.get('typescript', 0):,}\n"
            report += f"  - JavaScript: {loc.get('javascript', 0):,}\n"
            report += f"- **Test Coverage**: {coverage.get('current', 0)}% (Target: {coverage.get('target', 95)}%)\n\n"
        
        # Security
        security = self.metrics.get("metrics", {}).get("security", {})
        if security:
            report += "### Security\n"
            vulns = security.get("vulnerabilities", {})
            report += f"- **Security Score**: {security.get('security_score', 100)}/100\n"
            report += f"- **Vulnerabilities**:\n"
            report += f"  - High: {vulns.get('high', 0)}\n"
            report += f"  - Medium: {vulns.get('medium', 0)}\n"
            report += f"  - Low: {vulns.get('low', 0)}\n"
            report += f"- **Last Scan**: {security.get('last_scan', 'Never')}\n\n"
        
        # Performance
        benchmarks = self.metrics.get("benchmarks", {})
        if benchmarks:
            report += "### Performance Benchmarks\n"
            for category, tests in benchmarks.items():
                category_name = category.replace("_", " ").title()
                report += f"- **{category_name}**:\n"
                for test_name, test_data in tests.items():
                    if isinstance(test_data, dict):
                        current = test_data.get("current", 0)
                        target = test_data.get("target", 0)
                        if current > 0:
                            performance = "✅" if current <= target else "⚠️"
                            test_display = test_name.replace("_", " ").title()
                            report += f"  - {test_display}: {current} (Target: ≤{target}) {performance}\n"
                report += "\n"
        
        # Repository Stats
        repo = self.metrics.get("metrics", {}).get("repository", {})
        if repo:
            report += "### Repository Statistics\n"
            report += f"- **Stars**: {repo.get('stars', 0):,}\n"
            report += f"- **Forks**: {repo.get('forks', 0):,}\n"
            report += f"- **Contributors**: {repo.get('total_contributors', 0)}\n"
            report += f"- **Open Issues**: {repo.get('open_issues', 0)}\n"
            report += f"- **Open PRs**: {repo.get('open_prs', 0)}\n\n"
        
        # Recommendations
        report += "## 🎯 Recommendations\n\n"
        
        recommendations = []
        
        if categories.get("code_quality", 0) < 80:
            recommendations.append("- Improve test coverage and code documentation")
        
        if categories.get("security", 0) < 90:
            recommendations.append("- Address security vulnerabilities")
            recommendations.append("- Update dependencies with known issues")
        
        if categories.get("performance", 0) < 80:
            recommendations.append("- Optimize slow operations")
            recommendations.append("- Review performance benchmarks")
        
        if categories.get("community", 0) < 50:
            recommendations.append("- Increase community engagement")
            recommendations.append("- Improve documentation for contributors")
        
        if not recommendations:
            recommendations.append("- Continue current excellent practices! 🎉")
        
        report += "\n".join(recommendations)
        
        return report
    
    def _generate_health_html(self, timestamp: str) -> str:
        """Generate health report in HTML format."""
        markdown_content = self._generate_health_markdown(timestamp)
        
        # Simple Markdown to HTML conversion
        html_content = markdown_content.replace("\n", "<br>\n")
        html_content = html_content.replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>")
        html_content = html_content.replace("**", "<strong>").replace("*", "<em>")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Causal Interface Gym - Health Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .excellent {{ color: #27ae60; }}
        .good {{ color: #f39c12; }}
        .fair {{ color: #e67e22; }}
        .poor {{ color: #e74c3c; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>"""
        
        return html
    
    def _generate_health_pdf(self, timestamp: str) -> str:
        """Generate health report in PDF format."""
        if not PDF_AVAILABLE:
            raise ImportError("reportlab is required for PDF generation")
        
        output_file = self.output_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(str(output_file), pagesize=letter)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
        )
        story.append(Paragraph("Causal Interface Gym - Health Report", title_style))
        story.append(Paragraph(f"Generated: {timestamp}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Overall health score
        health = self.metrics.get("health", {})
        overall_score = health.get("overall_score", 0)
        
        story.append(Paragraph(f"Overall Health Score: {overall_score:.1f}/100", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Categories table
        categories = health.get("categories", {})
        if categories:
            data = [['Category', 'Score', 'Status']]
            for category, score in categories.items():
                status = "Good" if score >= 80 else "Fair" if score >= 60 else "Poor"
                category_name = category.replace("_", " ").title()
                data.append([category_name, f"{score:.1f}", status])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
        
        doc.build(story)
        return str(output_file)
    
    def generate_security_report(self, format_type: str = "markdown") -> str:
        """Generate security report."""
        security = self.metrics.get("metrics", {}).get("security", {})
        vulns = security.get("vulnerabilities", {})
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Security Report - Causal Interface Gym

*Generated: {timestamp}*

## Security Score: {security.get('security_score', 100)}/100

### Vulnerability Summary

- **Critical**: {vulns.get('critical', 0)}
- **High**: {vulns.get('high', 0)}
- **Medium**: {vulns.get('medium', 0)}
- **Low**: {vulns.get('low', 0)}

### Last Security Scan
{security.get('last_scan', 'Never performed')}

### Dependencies Status

"""
        
        deps = self.metrics.get("metrics", {}).get("dependencies", {})
        for lang, dep_info in deps.items():
            if isinstance(dep_info, dict):
                total = dep_info.get("total", 0)
                vulnerable = dep_info.get("vulnerable", 0)
                outdated = dep_info.get("outdated", 0)
                
                report += f"**{lang.title()}**:\n"
                report += f"- Total packages: {total}\n"
                report += f"- Vulnerable: {vulnerable}\n"
                report += f"- Outdated: {outdated}\n\n"
        
        # Security recommendations
        report += "## Recommendations\n\n"
        
        if vulns.get("high", 0) > 0 or vulns.get("critical", 0) > 0:
            report += "🚨 **URGENT**: Address high/critical vulnerabilities immediately\n"
        
        if vulns.get("medium", 0) > 5:
            report += "⚠️ **HIGH PRIORITY**: Review and fix medium-severity issues\n"
        
        for lang, dep_info in deps.items():
            if isinstance(dep_info, dict) and dep_info.get("vulnerable", 0) > 0:
                report += f"- Update vulnerable {lang} dependencies\n"
        
        return report
    
    def generate_performance_report(self, period: str = "weekly") -> str:
        """Generate performance report."""
        benchmarks = self.metrics.get("benchmarks", {})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Performance Report - Causal Interface Gym

*Generated: {timestamp} | Period: {period}*

## Performance Summary

"""
        
        for category, tests in benchmarks.items():
            category_name = category.replace("_", " ").title()
            report += f"### {category_name}\n\n"
            
            for test_name, test_data in tests.items():
                if isinstance(test_data, dict):
                    current = test_data.get("current", 0)
                    target = test_data.get("target", 0)
                    trend = test_data.get("trend", "stable")
                    
                    status = "✅" if current <= target else "⚠️"
                    trend_emoji = {"improving": "📈", "stable": "➡️", "declining": "📉"}.get(trend, "➡️")
                    
                    test_display = test_name.replace("_", " ").title()
                    report += f"**{test_display}**: {current} / {target} {status} {trend_emoji}\n"
            
            report += "\n"
        
        # Performance recommendations
        report += "## Performance Analysis\n\n"
        
        slow_tests = []
        for category, tests in benchmarks.items():
            for test_name, test_data in tests.items():
                if isinstance(test_data, dict):
                    current = test_data.get("current", 0)
                    target = test_data.get("target", 0)
                    if current > target * 1.2:  # 20% over target
                        slow_tests.append(f"{category}.{test_name}")
        
        if slow_tests:
            report += "**Performance Issues Detected**:\n"
            for test in slow_tests:
                report += f"- {test} is significantly over target\n"
        else:
            report += "✅ All performance benchmarks are within acceptable ranges.\n"
        
        return report
    
    def save_report(self, content: str, filename: str, format_type: str) -> str:
        """Save report to file."""
        if format_type == "html":
            file_path = self.output_dir / f"{filename}.html"
        elif format_type == "pdf":
            return content  # PDF is already saved in generation
        else:
            file_path = self.output_dir / f"{filename}.md"
        
        with open(file_path, "w") as f:
            f.write(content)
        
        return str(file_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate project reports")
    parser.add_argument("--type", choices=["health", "security", "performance"], 
                       default="health", help="Type of report to generate")
    parser.add_argument("--format", choices=["markdown", "html", "pdf"], 
                       default="markdown", help="Output format")
    parser.add_argument("--period", choices=["daily", "weekly", "monthly"], 
                       default="weekly", help="Reporting period")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    generator = ReportGenerator()
    
    try:
        if args.type == "health":
            content = generator.generate_health_report(args.format)
            filename = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif args.type == "security":
            content = generator.generate_security_report(args.format)
            filename = f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif args.type == "performance":
            content = generator.generate_performance_report(args.period)
            filename = f"performance_report_{args.period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                f.write(content)
            print(f"📄 Report saved to {output_path}")
        else:
            saved_path = generator.save_report(content, filename, args.format)
            print(f"📄 Report saved to {saved_path}")
        
        # Also print to stdout for immediate viewing
        if args.format == "markdown":
            print("\n" + "="*80)
            print(content)
    
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())