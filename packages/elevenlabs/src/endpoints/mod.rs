pub(crate) use crate::client::Result;
#[allow(unused_imports)]
pub(crate) use crate::shared::response_bodies::*;
pub use crate::shared::{PathParam, url::AndPathParam};
pub(crate) use bytes::Bytes;
pub(crate) use reqwest::{
    Method, Response, Url,
    multipart::{Form, Part},
};
pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use serde_json::Value;
// pub(crate) use base64; // Temporarily disabled

// Feature flags removed as they were unexpected
pub mod admin;
pub mod convai;
pub mod genai;

type QueryValues = Vec<(&'static str, String)>;

#[derive(Debug)]
pub enum RequestBody {
    Json(Value),
    Multipart(Form),
    Empty,
}

#[allow(async_fn_in_trait)]
pub trait ElevenLabsEndpoint {
    const BASE_URL: &'static str = "https://api.elevenlabs.io";

    const PATH: &'static str;

    const METHOD: Method;

    type ResponseBody;

    fn query_params(&self) -> Option<QueryValues> {
        None
    }

    fn path_params(&self) -> Vec<(&'static str, &str)> {
        vec![]
    }

    async fn request_body(&self) -> Result<RequestBody> {
        Ok(RequestBody::Empty)
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody>;

    fn url(&self) -> Result<Url> {
        let mut url = Self::BASE_URL
            .parse::<Url>()
            .map_err(|e| anyhow::anyhow!("Failed to parse base URL '{}': {}", Self::BASE_URL, e))?;

        let mut path = Self::PATH.to_string();

        for (placeholder, id) in self.path_params() {
            path = path.replace(placeholder, id);
        }

        url.set_path(&path);

        if let Some(query_params) = self.query_params() {
            url.query_pairs_mut().extend_pairs(query_params);
        }

        Ok(url)
    }
}
