#!/usr/bin/env python3
"""
TEI XML Parser for GROBID-processed papers.
Extracts structured information from TEI format XML files.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from logs import log


class TEIXMLParser:
    """Parser for GROBID TEI XML format papers."""

    # XML namespaces used in TEI
    NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def __init__(self):
        """Initialize TEI XML parser."""
        log.info("Initialized TEI XML parser")

    def parse_file(self, xml_path: str) -> Optional[Dict]:
        """
        Parse a TEI XML file and extract paper information.

        Args:
            xml_path: Path to TEI XML file

        Returns:
            Dictionary with paper metadata and content, or None if parsing fails
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract paper ID from filename
            paper_id = Path(xml_path).stem.replace('.grobid.tei', '')

            # Extract metadata from header
            title = self._extract_title(root)
            abstract = self._extract_abstract(root)
            authors = self._extract_authors(root)
            keywords = self._extract_keywords(root)

            # Extract main content from body
            content = self._extract_body_text(root)

            # Extract references (optional, can be used for citation analysis)
            references = self._extract_references(root)

            # Validate required fields
            if not title and not abstract:
                log.warning(f"Skipping {paper_id}: missing both title and abstract")
                return None

            # Extract or infer publish date from arXiv ID (format: YYMM.NNNNN)
            publish_date = self._extract_or_infer_date(root, paper_id)

            paper_data = {
                'paper_id': paper_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'categories': keywords,  # Using keywords as categories
                'content': content,
                'references': references,
                'publish_date': publish_date
            }

            log.debug(f"Parsed paper {paper_id}: {len(content)} chars, {len(authors)} authors")
            return paper_data

        except Exception as e:
            log.error(f"Error parsing {xml_path}: {e}")
            return None

    def _extract_title(self, root: ET.Element) -> str:
        """Extract paper title from TEI header."""
        title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', self.NS)
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
        return ""

    def _extract_abstract(self, root: ET.Element) -> str:
        """Extract abstract from TEI profileDesc."""
        abstract_elem = root.find('.//tei:profileDesc/tei:abstract', self.NS)
        if abstract_elem is not None:
            # Get all text from abstract, preserving structure
            return self._get_element_text(abstract_elem).strip()
        return ""

    def _extract_authors(self, root: ET.Element) -> List[str]:
        """Extract author names from TEI header."""
        authors = []

        # Look for authors in analytic section (paper-specific)
        author_elems = root.findall('.//tei:sourceDesc//tei:analytic//tei:author', self.NS)

        for author_elem in author_elems:
            # Extract forename and surname
            forename_elem = author_elem.find('.//tei:forename[@type="first"]', self.NS)
            surname_elem = author_elem.find('.//tei:surname', self.NS)

            forename = forename_elem.text.strip() if forename_elem is not None and forename_elem.text else ""
            surname = surname_elem.text.strip() if surname_elem is not None and surname_elem.text else ""

            if forename or surname:
                full_name = f"{forename} {surname}".strip()
                authors.append(full_name)

        return authors

    def _extract_keywords(self, root: ET.Element) -> List[str]:
        """Extract keywords from TEI textClass."""
        keywords = []
        keyword_elems = root.findall('.//tei:profileDesc/tei:textClass/tei:keywords/tei:term', self.NS)

        for keyword_elem in keyword_elems:
            if keyword_elem.text:
                keywords.append(keyword_elem.text.strip())

        return keywords

    def _extract_body_text(self, root: ET.Element) -> str:
        """Extract main body text with section structure preserved."""
        body_elem = root.find('.//tei:text/tei:body', self.NS)

        if body_elem is None:
            return ""

        sections = []
        div_elems = body_elem.findall('.//tei:div', self.NS)

        for div_elem in div_elems:
            # Extract section heading
            head_elem = div_elem.find('./tei:head', self.NS)
            if head_elem is not None and head_elem.text:
                section_title = head_elem.text.strip()
                sections.append(f"\n## {section_title}\n")

            # Extract section content (all paragraphs)
            para_elems = div_elem.findall('./tei:p', self.NS)
            for para_elem in para_elems:
                para_text = self._get_element_text(para_elem).strip()
                if para_text:
                    sections.append(para_text)

        # Join all sections
        content = '\n\n'.join(sections).strip()

        # Fallback: if no structured divs, extract all text
        if not content:
            content = self._get_element_text(body_elem).strip()

        return content

    def _extract_references(self, root: ET.Element) -> List[Dict]:
        """Extract bibliography references (optional)."""
        references = []
        bibl_elems = root.findall('.//tei:back//tei:listBibl/tei:biblStruct', self.NS)

        for i, bibl_elem in enumerate(bibl_elems):
            # Extract basic reference info
            ref_id = bibl_elem.get('{http://www.w3.org/XML/1998/namespace}id', f'ref_{i}')
            title_elem = bibl_elem.find('.//tei:title', self.NS)

            ref = {
                'ref_id': ref_id,
                'title': title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            }
            references.append(ref)

        return references

    def _get_element_text(self, element: ET.Element) -> str:
        """
        Recursively extract all text from an element and its children.
        Preserves text structure by joining with spaces.
        """
        texts = []

        # Get element's own text
        if element.text:
            texts.append(element.text)

        # Recursively get text from all children
        for child in element:
            child_text = self._get_element_text(child)
            if child_text:
                texts.append(child_text)

            # Get tail text (text after child element)
            if child.tail:
                texts.append(child.tail)

        return ' '.join(texts).strip()

    def _extract_or_infer_date(self, root: ET.Element, paper_id: str) -> str:
        """
        Extract publication date from TEI or infer from arXiv ID.
        arXiv ID format: YYMM.NNNNN (e.g., 0704.2083 = April 2007)
        """
        import re

        # Try to extract from XML first
        date_elem = root.find('.//tei:monogr/tei:imprint/tei:date', self.NS)
        if date_elem is not None and date_elem.get('when'):
            date_str = date_elem.get('when')
            # Validate it's a proper date format
            if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
                return date_str

        if date_elem is not None and date_elem.text and date_elem.text.strip():
            date_str = date_elem.text.strip()
            # Try to extract year from text like "2023 2019 2020" or just "2023"
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                year = year_match.group(0)
                return f"{year}-01-01"

        # Infer from arXiv ID format: YYMM.NNNNN
        match = re.match(r'(\d{2})(\d{2})\.', paper_id)
        if match:
            yy, mm = match.groups()
            # Convert YY to full year (assuming 2000+ for 00-99)
            year = int(yy)
            year = 1900 + year if year >= 91 else 2000 + year
            month = int(mm)
            # Validate month
            if 1 <= month <= 12:
                return f"{year}-{month:02d}-01"
            else:
                return f"{year}-01-01"

        return None


def main():
    """Test TEI XML parser."""
    parser = TEIXMLParser()

    # Test on sample file
    test_file = "/Users/admin/code/cazoodle/paper-search-engine/0704.2083.grobid.tei.xml"

    paper_data = parser.parse_file(test_file)

    if paper_data:
        print(f"Paper ID: {paper_data['paper_id']}")
        print(f"Title: {paper_data['title']}")
        print(f"Authors: {', '.join(paper_data['authors'])}")
        print(f"Keywords: {', '.join(paper_data['categories'])}")
        print(f"Abstract length: {len(paper_data['abstract'])} chars")
        print(f"Content length: {len(paper_data['content'])} chars")
        print(f"\nAbstract preview:\n{paper_data['abstract'][:200]}...")
    else:
        print("Failed to parse paper")


if __name__ == "__main__":
    main()
