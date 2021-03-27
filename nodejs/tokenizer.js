class Tokenizer {
  constructor(config = {}) {
    this.filters = config.filters || '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n';
    this.num_words = parseInt(config.num_words) || 0;
    this.word_index = config.word_index ? JSON.parse(config.word_index) : {};
    this.index_word = config.index_word ? JSON.parse(config.index_word) : {};
    this.word_counts = config.word_counts ? JSON.parse(config.word_counts) : {};
  }

  cleanText(text) {
    if (this.lower) text = text.toLowerCase();
    return text
      .replace(this.filters, '')
      .replace(/\s{2,}/g, ' ')
      .split(' ');
  }

  textsToSequences(texts) {
    return texts.map((text) =>
      this.cleanText(text).flatMap((word) =>
        this.word_index[word] &&
        (this.num_words === null || this.word_index[word] < this.num_words)
          ? this.word_index[word]
          : this.oov_token
          ? 1
          : []
      )
    );
  }
}

module.exports = { Tokenizer };
