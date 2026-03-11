use regex::Regex;

/// Normalizes raw document text for consistent chunking.
pub struct TextNormalizer {
    /// Compiled regex for collapsing whitespace
    whitespace_re: Regex,
    /// Compiled regex for stripping control characters (except newlines)
    control_re: Regex,
}

impl TextNormalizer {
    pub fn new() -> Self {
        Self {
            whitespace_re: Regex::new(r"[ \t]+").unwrap(),
            control_re: Regex::new(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]").unwrap(),
        }
    }

    /// Normalize text: collapse whitespace, strip control chars, normalize line endings.
    pub fn normalize(&self, text: &str) -> String {
        // Normalize line endings to \n
        let text = text.replace("\r\n", "\n").replace('\r', "\n");

        // Strip control characters (keep \n and \t temporarily)
        let text = self.control_re.replace_all(&text, "");

        // Collapse horizontal whitespace (spaces and tabs → single space)
        let text = self.whitespace_re.replace_all(&text, " ");

        // Collapse multiple blank lines into a single blank line
        let lines: Vec<&str> = text.lines().collect();
        let mut result = Vec::new();
        let mut prev_blank = false;
        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                if !prev_blank {
                    result.push("");
                    prev_blank = true;
                }
            } else {
                result.push(trimmed);
                prev_blank = false;
            }
        }

        result.join("\n").trim().to_string()
    }
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_collapse() {
        let n = TextNormalizer::new();
        assert_eq!(n.normalize("hello    world"), "hello world");
    }

    #[test]
    fn test_line_endings() {
        let n = TextNormalizer::new();
        let input = "line1\r\nline2\rline3\nline4";
        let result = n.normalize(input);
        assert!(result.contains("line1\nline2\nline3\nline4"));
    }

    #[test]
    fn test_blank_line_collapse() {
        let n = TextNormalizer::new();
        let input = "para1\n\n\n\npara2";
        let result = n.normalize(input);
        assert_eq!(result, "para1\n\npara2");
    }
}
