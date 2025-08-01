//! The audio native endpoint
use super::*;

/// Creates AudioNative enabled project, optionally starts conversion and returns project id and embeddable html snippet.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AudioNative {
    body: AudioNativeBody,
}

#[allow(dead_code)]
impl AudioNative {
    pub fn new(body: AudioNativeBody) -> Self {
        AudioNative { body }
    }
}

impl ElevenLabsEndpoint for AudioNative {
    const PATH: &'static str = "/v1/audio-native";

    const METHOD: Method = Method::POST;

    type ResponseBody = AudioNativeResponseBody;

    async fn request_body(&self) -> Result<RequestBody> {
        Ok(RequestBody::Multipart(self.body.clone().into()))
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.json().await?)
    }
}

#[derive(Clone, Debug, Default)]
pub struct AudioNativeBody {
    name: String,
    image: Option<String>,
    author: Option<String>,
    title: Option<String>,
    small: Option<bool>,
    text_color: Option<String>,
    background_color: Option<String>,
    sessionization: Option<u32>,
    voice_id: Option<String>,
    model_id: Option<String>,
    file: Option<String>,
    auto_convert: Option<bool>,
}

impl AudioNativeBody {
    pub fn new(name: &str) -> Self {
        AudioNativeBody {
            name: name.to_string(),
            ..Default::default()
        }
    }
    pub fn with_image(mut self, image: &str) -> Self {
        self.image = Some(image.to_string());
        self
    }
    pub fn with_author(mut self, author: &str) -> Self {
        self.author = Some(author.to_string());
        self
    }
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }
    pub fn with_small(mut self, small: bool) -> Self {
        self.small = Some(small);
        self
    }
    pub fn with_text_color(mut self, text_color: &str) -> Self {
        self.text_color = Some(text_color.to_string());
        self
    }
    pub fn with_background_color(mut self, background_color: &str) -> Self {
        self.background_color = Some(background_color.to_string());
        self
    }
    pub fn with_sessionization(mut self, sessionization: u32) -> Self {
        self.sessionization = Some(sessionization);
        self
    }
    pub fn with_voice_id(mut self, voice_id: &str) -> Self {
        self.voice_id = Some(voice_id.to_string());
        self
    }
    pub fn with_model_id(mut self, model_id: &str) -> Self {
        self.model_id = Some(model_id.to_string());
        self
    }
    pub fn with_file(mut self, file: &str) -> Self {
        self.file = Some(file.to_string());
        self
    }
    pub fn with_auto_convert(mut self) -> Self {
        self.auto_convert = Some(true);
        self
    }
}

impl From<AudioNativeBody> for Form {
    fn from(body: AudioNativeBody) -> Self {
        let mut form = Form::new();
        form = form.text("name", body.name);
        if let Some(image) = body.image {
            form = form.text("image", image);
        }
        if let Some(author) = body.author {
            form = form.text("author", author);
        }
        if let Some(title) = body.title {
            form = form.text("title", title);
        }
        if let Some(small) = body.small {
            form = form.text("small", small.to_string());
        }
        if let Some(text_color) = body.text_color {
            form = form.text("text_color", text_color);
        }
        if let Some(background_color) = body.background_color {
            form = form.text("background_color", background_color);
        }
        if let Some(sessionization) = body.sessionization {
            form = form.text("sessionization", sessionization.to_string());
        }
        if let Some(voice_id) = body.voice_id {
            form = form.text("voice_id", voice_id);
        }
        if let Some(model_id) = body.model_id {
            form = form.text("model_id", model_id);
        }
        if let Some(file) = body.file {
            form = form.text("file", file);
        }
        if let Some(auto_convert) = body.auto_convert {
            form = form.text("auto_convert", auto_convert.to_string());
        }
        form
    }
}

#[derive(Clone, Debug, Deserialize)]
#[allow(dead_code)]
pub struct AudioNativeResponseBody {
    pub project_id: String,
    #[allow(dead_code)]
    pub converting: bool,
    #[allow(dead_code)]
    pub html_snippet: String,
}
