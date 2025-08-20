"""Autonomous research paper generation system for causal interface gym findings."""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime
import uuid
import re
from pathlib import Path
import subprocess
import tempfile
import os

from ..llm.client import LLMClient
from .model_orchestrator import MultiAgentOrchestrator, AgentRole
from .causal_discovery_ai import QuantumCausalDiscovery, CausalHypothesis
from .experiment_evolution import ExperimentEvolution

logger = logging.getLogger(__name__)


class PaperSection(Enum):
    """Sections of a research paper."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"


class PaperType(Enum):
    """Types of research papers."""
    CONFERENCE = "conference"
    JOURNAL = "journal"
    WORKSHOP = "workshop"
    ARXIV = "arxiv"
    THESIS = "thesis"


@dataclass
class ResearchFinding:
    """Represents a research finding that can be included in papers."""
    finding_id: str
    title: str
    description: str
    methodology: str
    results: Dict[str, Any]
    statistical_significance: Dict[str, float]
    figures: List[str] = field(default_factory=list)
    related_work: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    novelty_score: float = 0.0
    impact_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PaperMetadata:
    """Metadata for generated research paper."""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    paper_type: PaperType
    target_venue: str
    word_count: int
    figures: List[str]
    tables: List[str]
    references: List[str]
    novelty_assessment: Dict[str, float]
    quality_scores: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GeneratedPaper:
    """Represents a complete generated research paper."""
    metadata: PaperMetadata
    sections: Dict[str, str]
    latex_source: str
    pdf_path: Optional[str] = None
    figures_generated: List[str] = field(default_factory=list)
    quality_assessment: Dict[str, float] = field(default_factory=dict)
    peer_review_feedback: List[Dict[str, Any]] = field(default_factory=list)


class ResearchPaperGenerator:
    """Autonomous system for generating research papers from experimental findings."""
    
    def __init__(self,
                 llm_orchestrator: MultiAgentOrchestrator,
                 output_directory: Path,
                 template_directory: Optional[Path] = None):
        """Initialize research paper generator.
        
        Args:
            llm_orchestrator: Multi-agent orchestrator for collaborative writing
            output_directory: Directory for generated papers
            template_directory: Directory containing LaTeX templates
        """
        self.llm_orchestrator = llm_orchestrator
        self.output_directory = Path(output_directory)
        self.template_directory = template_directory or (Path(__file__).parent / "templates")
        
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Research knowledge base
        self.research_findings: Dict[str, ResearchFinding] = {}
        self.generated_papers: Dict[str, GeneratedPaper] = {}
        self.literature_database: Dict[str, Any] = {}
        
        # Writing templates and styles
        self.paper_templates = self._load_paper_templates()
        self.writing_styles = self._initialize_writing_styles()
        
        # Specialized writing agents
        self.writing_agents = {
            AgentRole.METHODOLOGIST: "Methodology and experimental design sections",
            AgentRole.SYNTHESIZER: "Results synthesis and discussion",
            AgentRole.CRITIC: "Critical analysis and limitations",
            AgentRole.DOMAIN_EXPERT: "Domain-specific insights and related work"
        }
        
        # Quality metrics
        self.generation_metrics = {
            "papers_generated": 0,
            "avg_novelty_score": 0.0,
            "avg_quality_score": 0.0,
            "successful_compilations": 0,
            "peer_reviews_conducted": 0
        }
    
    def _load_paper_templates(self) -> Dict[str, str]:
        """Load LaTeX templates for different paper types."""
        templates = {}
        
        # Default conference template
        templates[PaperType.CONFERENCE.value] = r"""
\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{array}
\usepackage{mdwmath}
\usepackage{mdwtab}
\usepackage{eqparbox}
\usepackage{url}
\usepackage{graphicx}
\usepackage{cite}

\begin{document}

\title{{{title}}}

\author{
{{authors}}
}

\maketitle

\begin{abstract}
{{abstract}}
\end{abstract}

{{content}}

\end{document}
"""
        
        # Journal template
        templates[PaperType.JOURNAL.value] = r"""
\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{array}
\usepackage{mdwmath}
\usepackage{mdwtab}
\usepackage{eqparbox}
\usepackage{url}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{hyperref}

\begin{document}

\title{{{title}}}

\author{{{authors}}}

\markboth{Journal of Causal Interface Research, Vol. X, No. Y, Month 2025}{}

\maketitle

\begin{abstract}
{{abstract}}
\end{abstract}

\begin{IEEEkeywords}
{{keywords}}
\end{IEEEkeywords}

{{content}}

\end{document}
"""
        
        return templates
    
    def _initialize_writing_styles(self) -> Dict[str, Dict[str, str]]:
        """Initialize writing style guidelines for different paper types."""
        return {
            "academic": {
                "tone": "formal, objective, precise",
                "sentence_structure": "complex but clear, passive voice where appropriate",
                "terminology": "domain-specific, well-defined",
                "citations": "extensive, authoritative sources"
            },
            "technical": {
                "tone": "precise, methodical, detailed",
                "sentence_structure": "clear, direct, step-by-step",
                "terminology": "technical accuracy paramount",
                "citations": "focus on methodology papers"
            },
            "accessible": {
                "tone": "engaging while maintaining rigor",
                "sentence_structure": "varied, some shorter sentences for clarity",
                "terminology": "explained technical terms",
                "citations": "mix of technical and accessible sources"
            }
        }
    
    async def register_research_finding(self,
                                      finding: ResearchFinding) -> str:
        """Register a research finding for potential inclusion in papers."""
        self.research_findings[finding.finding_id] = finding
        
        # Analyze novelty and impact
        await self._analyze_finding_novelty(finding)
        await self._assess_finding_impact(finding)
        
        logger.info(f"Registered research finding: {finding.title}")
        return finding.finding_id
    
    async def _analyze_finding_novelty(self, finding: ResearchFinding) -> float:
        """Analyze the novelty of a research finding."""
        # In practice, would search literature databases
        # For now, use heuristics based on methodology and results
        
        novelty_factors = []
        
        # Methodology novelty
        novel_methods = ["quantum", "federated", "autonomous", "multi-agent"]
        methodology_novelty = sum(1 for method in novel_methods 
                                if method in finding.methodology.lower()) / len(novel_methods)
        novelty_factors.append(methodology_novelty)
        
        # Results significance
        if finding.statistical_significance:
            p_values = [v for v in finding.statistical_significance.values() if v < 0.05]
            significance_novelty = len(p_values) / max(len(finding.statistical_significance), 1)
            novelty_factors.append(significance_novelty)
        
        # Overall novelty score
        finding.novelty_score = np.mean(novelty_factors) if novelty_factors else 0.5
        return finding.novelty_score
    
    async def _assess_finding_impact(self, finding: ResearchFinding) -> float:
        """Assess potential impact of a research finding."""
        impact_factors = []
        
        # Performance improvements
        if "improvement" in finding.description.lower():
            # Look for percentage improvements
            improvement_matches = re.findall(r'(\d+(?:\.\d+)?)%', finding.description)
            if improvement_matches:
                max_improvement = max(float(match) for match in improvement_matches)
                impact_factors.append(min(max_improvement / 100.0, 1.0))
        
        # Practical applicability
        practical_terms = ["deployment", "production", "scalable", "efficient"]
        practical_score = sum(1 for term in practical_terms 
                            if term in finding.description.lower()) / len(practical_terms)
        impact_factors.append(practical_score)
        
        # Theoretical significance
        theoretical_terms = ["theory", "framework", "model", "algorithm"]
        theoretical_score = sum(1 for term in theoretical_terms 
                              if term in finding.methodology.lower()) / len(theoretical_terms)
        impact_factors.append(theoretical_score)
        
        finding.impact_score = np.mean(impact_factors) if impact_factors else 0.5
        return finding.impact_score
    
    async def generate_research_paper(self,
                                    paper_title: str,
                                    research_question: str,
                                    target_findings: List[str],
                                    paper_type: PaperType = PaperType.CONFERENCE,
                                    target_venue: str = "International Conference on Causal AI",
                                    authors: Optional[List[str]] = None) -> str:
        """Generate a complete research paper from findings.
        
        Args:
            paper_title: Title of the paper
            research_question: Main research question
            target_findings: List of finding IDs to include
            paper_type: Type of paper to generate
            target_venue: Target publication venue
            authors: List of authors
            
        Returns:
            Paper ID
        """
        paper_id = f"paper_{uuid.uuid4().hex[:8]}"
        
        # Validate findings
        selected_findings = []
        for finding_id in target_findings:
            if finding_id in self.research_findings:
                selected_findings.append(self.research_findings[finding_id])
            else:
                logger.warning(f"Finding {finding_id} not found")
        
        if not selected_findings:
            raise ValueError("No valid findings provided")
        
        # Generate paper sections collaboratively
        sections = await self._generate_paper_sections(
            paper_title, research_question, selected_findings, paper_type
        )
        
        # Generate figures and tables
        figures = await self._generate_figures(selected_findings, paper_id)
        
        # Create metadata
        metadata = PaperMetadata(
            paper_id=paper_id,
            title=paper_title,
            authors=authors or ["AI Research Assistant", "Autonomous System"],
            abstract=sections.get(PaperSection.ABSTRACT.value, ""),
            keywords=await self._extract_keywords(sections),
            paper_type=paper_type,
            target_venue=target_venue,
            word_count=sum(len(text.split()) for text in sections.values()),
            figures=figures,
            tables=[],  # Would generate tables in full implementation
            references=await self._generate_references(selected_findings),
            novelty_assessment=await self._assess_paper_novelty(selected_findings),
            quality_scores={}
        )
        
        # Generate LaTeX source
        latex_source = await self._compile_latex_document(metadata, sections)
        
        # Create paper object
        paper = GeneratedPaper(
            metadata=metadata,
            sections=sections,
            latex_source=latex_source,
            figures_generated=figures
        )
        
        # Compile to PDF
        pdf_path = await self._compile_to_pdf(paper)
        if pdf_path:
            paper.pdf_path = str(pdf_path)
        
        # Quality assessment
        paper.quality_assessment = await self._assess_paper_quality(paper)
        
        # Store paper
        self.generated_papers[paper_id] = paper
        
        # Update metrics
        self.generation_metrics["papers_generated"] += 1
        if pdf_path:
            self.generation_metrics["successful_compilations"] += 1
        
        logger.info(f"Generated research paper: {paper_title}")
        return paper_id
    
    async def _generate_paper_sections(self,
                                     title: str,
                                     research_question: str,
                                     findings: List[ResearchFinding],
                                     paper_type: PaperType) -> Dict[str, str]:
        """Generate all sections of the paper collaboratively."""
        sections = {}
        
        # Create collaborative task for paper generation
        paper_context = {
            "title": title,
            "research_question": research_question,
            "findings": [
                {
                    "title": f.title,
                    "methodology": f.methodology,
                    "results": f.results,
                    "implications": f.implications
                }
                for f in findings
            ],
            "paper_type": paper_type.value,
            "style_guidelines": self.writing_styles["academic"]
        }
        
        # Generate each section
        for section in PaperSection:
            if section != PaperSection.REFERENCES:  # References handled separately
                section_content = await self._generate_section(
                    section, paper_context, findings
                )
                sections[section.value] = section_content
        
        return sections
    
    async def _generate_section(self,
                              section: PaperSection,
                              context: Dict[str, Any],
                              findings: List[ResearchFinding]) -> str:
        """Generate a specific section of the paper."""
        # Section-specific prompts
        section_prompts = {
            PaperSection.ABSTRACT: """
            Write a comprehensive abstract for this research paper. Include:
            1. Problem statement and motivation
            2. Methodology overview
            3. Key findings and contributions
            4. Implications and significance
            Keep it concise (150-250 words) but informative.
            """,
            
            PaperSection.INTRODUCTION: """
            Write an engaging introduction that:
            1. Establishes the research problem and its importance
            2. Provides necessary background context
            3. Clearly states the research question and objectives
            4. Outlines the paper structure
            Use a funnel approach: broad context -> specific problem -> our approach.
            """,
            
            PaperSection.METHODOLOGY: """
            Describe the methodology with sufficient detail for reproduction:
            1. Overall approach and framework
            2. Experimental design and setup
            3. Data collection and preprocessing
            4. Algorithms and techniques used
            5. Evaluation metrics and validation methods
            Be precise and technical while maintaining clarity.
            """,
            
            PaperSection.RESULTS: """
            Present the results clearly and objectively:
            1. Main findings with statistical support
            2. Performance comparisons and benchmarks
            3. Analysis of different conditions or parameters
            4. Visualization references (figures and tables)
            Focus on facts, save interpretation for discussion.
            """,
            
            PaperSection.DISCUSSION: """
            Provide insightful discussion of the results:
            1. Interpretation of findings in context
            2. Comparison with related work
            3. Implications for theory and practice
            4. Limitations and potential confounding factors
            5. Future research directions
            Be analytical and balanced in your assessment.
            """,
            
            PaperSection.CONCLUSION: """
            Write a strong conclusion that:
            1. Summarizes key contributions
            2. Restates the significance of findings
            3. Discusses broader implications
            4. Suggests concrete next steps
            End with impact - why should readers care about this work?
            """
        }
        
        prompt = section_prompts.get(section, "Write this section of the research paper.")
        
        # Use collaborative writing approach
        writing_task = {
            "section": section.value,
            "context": context,
            "findings_data": [
                {
                    "title": f.title,
                    "description": f.description,
                    "methodology": f.methodology,
                    "results": f.results,
                    "statistical_significance": f.statistical_significance,
                    "implications": f.implications
                }
                for f in findings
            ],
            "prompt": prompt,
            "style": context.get("style_guidelines", {})
        }
        
        # Get collaborative input from multiple agents
        collaboration_result = await self.llm_orchestrator.orchestrate_causal_analysis(
            writing_task,
            required_roles=[AgentRole.SYNTHESIZER, AgentRole.METHODOLOGIST, AgentRole.CRITIC]
        )
        
        if collaboration_result.get("status") == "success":
            section_content = collaboration_result.get("synthesis", {}).get("response", "")
            return section_content
        else:
            # Fallback to simple generation
            return f"Section {section.value} content would be generated here."
    
    async def _generate_figures(self,
                              findings: List[ResearchFinding],
                              paper_id: str) -> List[str]:
        """Generate figures for the research paper."""
        figures = []
        figure_dir = self.output_directory / paper_id / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate performance comparison figures
        for i, finding in enumerate(findings[:3]):  # Limit to first 3 findings
            if finding.results and isinstance(finding.results, dict):
                figure_path = await self._create_performance_figure(
                    finding, figure_dir, f"figure_{i+1}"
                )
                if figure_path:
                    figures.append(str(figure_path))
        
        # Generate methodology diagram
        methodology_figure = await self._create_methodology_diagram(
            findings, figure_dir, "methodology_overview"
        )
        if methodology_figure:
            figures.append(str(methodology_figure))
        
        return figures
    
    async def _create_performance_figure(self,
                                       finding: ResearchFinding,
                                       output_dir: Path,
                                       filename: str) -> Optional[Path]:
        """Create a performance comparison figure."""
        try:
            # Extract performance metrics
            results = finding.results
            if not results:
                return None
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Example: Bar chart of performance metrics
            metrics = []
            values = []
            
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    metrics.append(key.replace("_", " ").title())
                    values.append(value)
            
            if metrics and values:
                plt.bar(metrics, values)
                plt.title(f"Performance Results: {finding.title}")
                plt.xlabel("Metrics")
                plt.ylabel("Values")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save figure
                figure_path = output_dir / f"{filename}.png"
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return figure_path
            
        except Exception as e:
            logger.error(f"Failed to create performance figure: {e}")
        
        return None
    
    async def _create_methodology_diagram(self,
                                        findings: List[ResearchFinding],
                                        output_dir: Path,
                                        filename: str) -> Optional[Path]:
        """Create a methodology overview diagram."""
        try:
            # Create a simple flowchart-style diagram
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define methodology steps
            steps = [
                "Data Collection",
                "Causal Discovery", 
                "Model Training",
                "Validation",
                "Results Analysis"
            ]
            
            # Create boxes and arrows
            y_positions = np.linspace(0.8, 0.2, len(steps))
            x_position = 0.5
            
            for i, (step, y_pos) in enumerate(zip(steps, y_positions)):
                # Draw box
                bbox = dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7)
                ax.text(x_position, y_pos, step, ha='center', va='center',
                       fontsize=12, bbox=bbox, transform=ax.transAxes)
                
                # Draw arrow to next step
                if i < len(steps) - 1:
                    ax.annotate('', xy=(x_position, y_positions[i+1] + 0.05),
                              xytext=(x_position, y_pos - 0.05),
                              arrowprops=dict(arrowstyle='->', lw=2, color='gray'),
                              transform=ax.transAxes)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title("Methodology Overview", fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Save figure
            figure_path = output_dir / f"{filename}.png"
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return figure_path
            
        except Exception as e:
            logger.error(f"Failed to create methodology diagram: {e}")
        
        return None
    
    async def _extract_keywords(self, sections: Dict[str, str]) -> List[str]:
        """Extract keywords from paper sections."""
        # Combine abstract and introduction for keyword extraction
        text = sections.get("abstract", "") + " " + sections.get("introduction", "")
        
        # Common causal inference keywords
        causal_keywords = [
            "causal inference", "causal discovery", "do-calculus", "intervention",
            "causal graph", "directed acyclic graph", "confounding", "backdoor",
            "instrumental variables", "causal effect", "causality"
        ]
        
        # AI/ML keywords
        ai_keywords = [
            "machine learning", "artificial intelligence", "neural networks",
            "deep learning", "reinforcement learning", "federated learning",
            "quantum computing", "multi-agent systems"
        ]
        
        # Extract keywords that appear in the text
        extracted_keywords = []
        text_lower = text.lower()
        
        for keyword in causal_keywords + ai_keywords:
            if keyword in text_lower:
                extracted_keywords.append(keyword)
        
        # Add domain-specific terms
        domain_terms = ["interface design", "user experience", "experimental design"]
        for term in domain_terms:
            if term in text_lower:
                extracted_keywords.append(term)
        
        return extracted_keywords[:8]  # Limit to 8 keywords
    
    async def _generate_references(self, findings: List[ResearchFinding]) -> List[str]:
        """Generate reference list for the paper."""
        references = []
        
        # Core causal inference references
        core_refs = [
            "J. Pearl, Causality: Models, Reasoning, and Inference, 2nd ed. Cambridge University Press, 2009.",
            "P. Spirtes, C. Glymour, and R. Scheines, Causation, Prediction, and Search, 2nd ed. MIT Press, 2000.",
            "M. A. Hernán and J. M. Robins, Causal Inference: What If. Chapman & Hall/CRC, 2020.",
            "E. Bareinboim and J. Pearl, 'Causal inference and the data-fusion problem,' Proc. Natl. Acad. Sci., vol. 113, no. 27, pp. 7345-7352, 2016."
        ]
        
        # Add methodology-specific references based on findings
        for finding in findings:
            methodology = finding.methodology.lower()
            
            if "quantum" in methodology:
                references.append(
                    "S. Lloyd, 'Quantum algorithms for supervised and unsupervised machine learning,' arXiv:1307.0411, 2013."
                )
            
            if "federated" in methodology:
                references.append(
                    "H. B. McMahan et al., 'Communication-efficient learning of deep networks from decentralized data,' AISTATS, 2017."
                )
            
            if "multi-agent" in methodology:
                references.append(
                    "M. Tampuu et al., 'Multiagent cooperation and competition with deep reinforcement learning,' PLoS ONE, 2017."
                )
        
        # Add recent related work
        references.extend([
            "T. Zečević et al., 'Causal parrots: Large language models may talk causality but are not causal,' arXiv:2308.13067, 2023.",
            "A. Kičiman et al., 'Causal reasoning and large language models: Opening a new frontier for causality,' arXiv:2305.00050, 2023."
        ])
        
        return core_refs + references[:10]  # Limit total references
    
    async def _assess_paper_novelty(self, findings: List[ResearchFinding]) -> Dict[str, float]:
        """Assess overall novelty of the paper."""
        if not findings:
            return {"overall_novelty": 0.0}
        
        # Aggregate novelty from individual findings
        finding_novelties = [f.novelty_score for f in findings]
        
        return {
            "overall_novelty": np.mean(finding_novelties),
            "max_finding_novelty": max(finding_novelties),
            "min_finding_novelty": min(finding_novelties),
            "novelty_variance": np.var(finding_novelties),
            "findings_above_threshold": sum(1 for n in finding_novelties if n > 0.7) / len(finding_novelties)
        }
    
    async def _compile_latex_document(self,
                                    metadata: PaperMetadata,
                                    sections: Dict[str, str]) -> str:
        """Compile sections into complete LaTeX document."""
        template = self.paper_templates.get(metadata.paper_type.value, 
                                          self.paper_templates[PaperType.CONFERENCE.value])
        
        # Prepare content sections
        content_parts = []
        
        # Order sections appropriately
        section_order = [
            PaperSection.INTRODUCTION,
            PaperSection.RELATED_WORK,
            PaperSection.METHODOLOGY,
            PaperSection.EXPERIMENTS,
            PaperSection.RESULTS,
            PaperSection.DISCUSSION,
            PaperSection.CONCLUSION
        ]
        
        for section in section_order:
            if section.value in sections and sections[section.value].strip():
                section_title = section.value.replace('_', ' ').title()
                content_parts.append(f"\\section{{{section_title}}}")
                content_parts.append(sections[section.value])
                content_parts.append("")  # Add spacing
        
        # Add references section
        if metadata.references:
            content_parts.append("\\section{References}")
            content_parts.append("\\begin{thebibliography}{99}")
            for i, ref in enumerate(metadata.references, 1):
                content_parts.append(f"\\bibitem{{ref{i}}} {ref}")
            content_parts.append("\\end{thebibliography}")
        
        content = "\n".join(content_parts)
        
        # Fill template
        latex_source = template.format(
            title=metadata.title,
            authors=" \\\\ ".join(metadata.authors),
            abstract=sections.get(PaperSection.ABSTRACT.value, ""),
            keywords=", ".join(metadata.keywords),
            content=content
        )
        
        return latex_source
    
    async def _compile_to_pdf(self, paper: GeneratedPaper) -> Optional[Path]:
        """Compile LaTeX source to PDF."""
        try:
            # Create temporary directory for compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write LaTeX source
                tex_file = temp_path / "paper.tex"
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(paper.latex_source)
                
                # Copy figures if they exist
                if paper.figures_generated:
                    figures_dir = temp_path / "figures"
                    figures_dir.mkdir(exist_ok=True)
                    
                    for figure_path in paper.figures_generated:
                        if Path(figure_path).exists():
                            import shutil
                            shutil.copy2(figure_path, figures_dir / Path(figure_path).name)
                
                # Compile LaTeX (requires pdflatex installation)
                try:
                    # First compilation
                    result = subprocess.run([
                        'pdflatex', '-interaction=nonstopmode', 'paper.tex'
                    ], cwd=temp_path, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        # Second compilation for references
                        subprocess.run([
                            'pdflatex', '-interaction=nonstopmode', 'paper.tex'
                        ], cwd=temp_path, capture_output=True, text=True, timeout=60)
                        
                        # Copy PDF to output directory
                        pdf_source = temp_path / "paper.pdf"
                        if pdf_source.exists():
                            paper_dir = self.output_directory / paper.metadata.paper_id
                            paper_dir.mkdir(parents=True, exist_ok=True)
                            
                            pdf_dest = paper_dir / f"{paper.metadata.paper_id}.pdf"
                            import shutil
                            shutil.copy2(pdf_source, pdf_dest)
                            
                            # Also save LaTeX source
                            tex_dest = paper_dir / f"{paper.metadata.paper_id}.tex"
                            with open(tex_dest, 'w', encoding='utf-8') as f:
                                f.write(paper.latex_source)
                            
                            return pdf_dest
                    else:
                        logger.error(f"LaTeX compilation failed: {result.stderr}")
                
                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    logger.warning(f"PDF compilation failed (LaTeX not available?): {e}")
                    # Save LaTeX source anyway
                    paper_dir = self.output_directory / paper.metadata.paper_id
                    paper_dir.mkdir(parents=True, exist_ok=True)
                    
                    tex_dest = paper_dir / f"{paper.metadata.paper_id}.tex"
                    with open(tex_dest, 'w', encoding='utf-8') as f:
                        f.write(paper.latex_source)
        
        except Exception as e:
            logger.error(f"PDF compilation error: {e}")
        
        return None
    
    async def _assess_paper_quality(self, paper: GeneratedPaper) -> Dict[str, float]:
        """Assess overall quality of generated paper."""
        quality_scores = {}
        
        # Content quality assessment
        sections = paper.sections
        
        # Word count appropriateness
        total_words = paper.metadata.word_count
        if paper.metadata.paper_type == PaperType.CONFERENCE:
            ideal_range = (3000, 6000)
        else:  # Journal
            ideal_range = (6000, 12000)
        
        word_score = 1.0 - abs(total_words - np.mean(ideal_range)) / np.mean(ideal_range)
        quality_scores["word_count_appropriateness"] = max(0, min(1, word_score))
        
        # Section completeness
        required_sections = [s.value for s in PaperSection if s != PaperSection.REFERENCES]
        present_sections = [s for s in required_sections if sections.get(s, "").strip()]
        quality_scores["section_completeness"] = len(present_sections) / len(required_sections)
        
        # Abstract quality (keywords present)
        abstract = sections.get("abstract", "")
        abstract_keywords = sum(1 for keyword in paper.metadata.keywords 
                              if keyword.lower() in abstract.lower())
        quality_scores["abstract_keyword_coverage"] = abstract_keywords / max(len(paper.metadata.keywords), 1)
        
        # Figure/table integration
        if paper.figures_generated:
            quality_scores["visual_content"] = 1.0
        else:
            quality_scores["visual_content"] = 0.0
        
        # Reference appropriateness
        ref_count = len(paper.metadata.references)
        if paper.metadata.paper_type == PaperType.CONFERENCE:
            ideal_refs = 25
        else:
            ideal_refs = 40
        
        ref_score = min(1.0, ref_count / ideal_refs)
        quality_scores["reference_coverage"] = ref_score
        
        # Overall quality
        quality_scores["overall_quality"] = np.mean(list(quality_scores.values()))
        
        return quality_scores
    
    async def conduct_ai_peer_review(self, paper_id: str) -> Dict[str, Any]:
        """Conduct AI-powered peer review of generated paper."""
        if paper_id not in self.generated_papers:
            raise ValueError(f"Paper {paper_id} not found")
        
        paper = self.generated_papers[paper_id]
        
        # Create peer review task
        review_context = {
            "paper_title": paper.metadata.title,
            "abstract": paper.sections.get("abstract", ""),
            "sections": paper.sections,
            "paper_type": paper.metadata.paper_type.value,
            "target_venue": paper.metadata.target_venue
        }
        
        # Use multiple critic agents for comprehensive review
        review_result = await self.llm_orchestrator.orchestrate_causal_analysis(
            {
                "task": "peer_review",
                "paper_content": review_context,
                "review_criteria": [
                    "novelty and significance",
                    "technical quality and rigor", 
                    "clarity and presentation",
                    "experimental validation",
                    "related work coverage",
                    "reproducibility"
                ]
            },
            required_roles=[AgentRole.CRITIC, AgentRole.METHODOLOGIST, AgentRole.DOMAIN_EXPERT]
        )
        
        # Extract review feedback
        feedback = {
            "overall_rating": 0.0,
            "strengths": [],
            "weaknesses": [],
            "detailed_comments": {},
            "recommendation": "unknown"
        }
        
        if review_result.get("status") == "success":
            synthesis = review_result.get("synthesis", {}).get("response", "")
            
            # Parse review content (simplified)
            if "accept" in synthesis.lower():
                feedback["recommendation"] = "accept"
                feedback["overall_rating"] = 0.8
            elif "reject" in synthesis.lower():
                feedback["recommendation"] = "reject" 
                feedback["overall_rating"] = 0.3
            else:
                feedback["recommendation"] = "revise"
                feedback["overall_rating"] = 0.6
            
            feedback["detailed_comments"]["ai_review"] = synthesis
        
        # Store review
        paper.peer_review_feedback.append({
            "reviewer": "AI_System",
            "timestamp": datetime.now(),
            **feedback
        })
        
        # Update metrics
        self.generation_metrics["peer_reviews_conducted"] += 1
        
        return feedback
    
    def get_paper_status(self, paper_id: str) -> Dict[str, Any]:
        """Get status and metadata for a generated paper."""
        if paper_id not in self.generated_papers:
            return {"error": "Paper not found"}
        
        paper = self.generated_papers[paper_id]
        
        return {
            "paper_id": paper_id,
            "title": paper.metadata.title,
            "authors": paper.metadata.authors,
            "paper_type": paper.metadata.paper_type.value,
            "word_count": paper.metadata.word_count,
            "sections_completed": len(paper.sections),
            "figures_generated": len(paper.figures_generated),
            "pdf_available": paper.pdf_path is not None,
            "quality_scores": paper.quality_assessment,
            "peer_reviews": len(paper.peer_review_feedback),
            "novelty_assessment": paper.metadata.novelty_assessment,
            "created_at": paper.metadata.created_at.isoformat()
        }
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of paper generation activities."""
        if self.generated_papers:
            avg_quality = np.mean([
                p.quality_assessment.get("overall_quality", 0.0) 
                for p in self.generated_papers.values()
            ])
            
            avg_novelty = np.mean([
                p.metadata.novelty_assessment.get("overall_novelty", 0.0)
                for p in self.generated_papers.values()
            ])
        else:
            avg_quality = 0.0
            avg_novelty = 0.0
        
        return {
            **self.generation_metrics,
            "avg_quality_score": avg_quality,
            "avg_novelty_score": avg_novelty,
            "research_findings_available": len(self.research_findings),
            "papers_generated": len(self.generated_papers),
            "recent_papers": [
                {
                    "paper_id": pid,
                    "title": paper.metadata.title,
                    "quality": paper.quality_assessment.get("overall_quality", 0.0),
                    "created": paper.metadata.created_at.isoformat()
                }
                for pid, paper in list(self.generated_papers.items())[-5:]
            ]
        }