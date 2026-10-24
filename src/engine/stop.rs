//! Stop-sequence matching over a stream of text pieces.
//!
//! A stop sequence can arrive split across several tokens, so the tail of the output that could
//! still turn into one is held back until it is clear whether it will.

pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
    longest: usize,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        let longest = stops.iter().map(String::len).max().unwrap_or(0);
        Self { stops, held: String::new(), longest }
    }

    /// Feeds the next piece of generated text. Returns the text that is safe to emit and
    /// whether a stop sequence was reached (the sequence itself is never emitted).
    pub fn push(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        self.held.push_str(text);

        if let Some(at) = self.earliest_stop() {
            let before = self.held[..at].to_string();
            self.held.clear();
            return (before, true);
        }

        let emit_len = self.held.len() - self.held_suffix_len();
        let emit = self.held[..emit_len].to_string();
        self.held.drain(..emit_len);
        (emit, false)
    }

    /// Returns whatever is still held back once generation ends without hitting a stop.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    fn earliest_stop(&self) -> Option<usize> {
        self.stops.iter().filter_map(|stop| self.held.find(stop.as_str())).min()
    }

    /// Length of the longest suffix of `held` that is a proper prefix of some stop sequence.
    fn held_suffix_len(&self) -> usize {
        let longest_possible = self.longest.saturating_sub(1).min(self.held.len());
        (1..=longest_possible)
            .rev()
            .find(|&len| {
                let start = self.held.len() - len;
                self.held.is_char_boundary(start)
                    && self
                        .stops
                        .iter()
                        .any(|stop| stop.len() > len && stop.starts_with(&self.held[start..]))
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(stops: &[&str]) -> StopMatcher {
        StopMatcher::new(stops.iter().map(|s| (*s).to_string()).collect())
    }

    /// Feeds `pieces` through a matcher and returns everything emitted plus whether it stopped.
    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut matcher = matcher(stops);
        let mut out = String::new();
        for piece in pieces {
            let (text, hit) = matcher.push(piece);
            out.push_str(&text);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&matcher.flush());
        (out, false)
    }

    #[test]
    fn without_stops_text_passes_straight_through() {
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), false));
    }

    #[test]
    fn stops_inside_one_piece_and_hides_the_sequence() {
        assert_eq!(run(&["END"], &["hello END world"]), ("hello ".into(), true));
    }

    #[test]
    fn stops_when_the_sequence_is_split_across_pieces() {
        assert_eq!(run(&["END"], &["hello E", "N", "D tail"]), ("hello ".into(), true));
    }

    #[test]
    fn releases_a_held_prefix_that_never_completed() {
        assert_eq!(
            run(&["END"], &["an E", "x-less ", "E", "nd"]),
            ("an Ex-less End".into(), false)
        );
    }

    #[test]
    fn picks_the_earliest_of_several_stops() {
        assert_eq!(run(&["two", "one"], &["zero one two"]), ("zero ".into(), true));
    }

    #[test]
    fn flush_returns_the_held_tail() {
        let mut matcher = matcher(&["STOP"]);
        assert_eq!(matcher.push("abc ST"), ("abc ".into(), false));
        assert_eq!(matcher.flush(), "ST");
    }

    #[test]
    fn handles_multibyte_text_without_splitting_characters() {
        assert_eq!(run(&["fin"], &["héllo wörld ", "f", "in"]), ("héllo wörld ".into(), true));
        assert_eq!(run(&["日本"], &["こんにちは日", "本語"]), ("こんにちは".into(), true));
    }
}
