//! Agents endpoints

use super::*;
use crate::endpoints::convai::knowledge_base::EmbeddingModel;
use crate::endpoints::convai::phone_numbers::{AssignedAgent, PhoneNumberProvider};
use crate::endpoints::convai::workspace::{ConversationInitiationClientDataWebhook, Webhooks};
use crate::shared::{AccessLevel, DictionaryLocator};
use std::collections::HashMap;

/// Create an agent from a config object
///
/// # Example
/// ```no_run
/// use speakrs_elevenlabs::{ElevenLabsClient, Result};
/// use speakrs_elevenlabs::endpoints::convai::agents::{
///     CreateAgent, CreateAgentBody, ConversationConfig, AgentConfig, PromptConfig};
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let client = ElevenLabsClient::from_env()?;
///
///     let prompt_config = PromptConfig::default().with_prompt("some_prompt");
///
///     let agent_config = AgentConfig::default().with_prompt(prompt_config);
///
///     let convo_config = ConversationConfig::default()
///      .with_agent_config(agent_config);
///
///     let body = CreateAgentBody::new(convo_config);
///
///     let endpoint = CreateAgent::new(body);
///
///     let resp = client.hit(endpoint).await?;
///
///     println!("{:?}", resp);
///
///     Ok(())
/// }
/// ```
/// See [Create Agent API reference](https://elevenlabs.io/docs/conversational-ai/api-reference/agents/create-agent)
#[derive(Clone, Debug)]
pub struct CreateAgent {
    body: CreateAgentBody,
    query: Option<AgentQuery>,
}

impl CreateAgent {
    pub fn new(body: CreateAgentBody) -> Self {
        CreateAgent { body, query: None }
    }

    pub fn with_query(mut self, query: AgentQuery) -> Self {
        self.query = Some(query);
        self
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CreateAgentBody {
    pub conversation_config: ConversationConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform_settings: Option<PlatformSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl CreateAgentBody {
    pub fn new(conversation_config: ConversationConfig) -> Self {
        CreateAgentBody {
            conversation_config,
            platform_settings: None,
            name: None,
        }
    }
    pub fn with_platform_settings(mut self, platform_settings: PlatformSettings) -> Self {
        self.platform_settings = Some(platform_settings);
        self
    }
}

impl ElevenLabsEndpoint for CreateAgent {
    const PATH: &'static str = "/v1/convai/agents/create";

    const METHOD: Method = Method::POST;

    type ResponseBody = CreateAgentResponse;

    fn query_params(&self) -> Option<QueryValues> {
        self.query.as_ref().map(|q| q.params.clone())
    }

    async fn request_body(&self) -> Result<RequestBody> {
        TryInto::try_into(&self.body)
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.json().await?)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CreateAgentResponse {
    pub agent_id: String,
}

impl TryFrom<&CreateAgentBody> for RequestBody {
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn try_from(body: &CreateAgentBody) -> Result<Self> {
        Ok(RequestBody::Json(serde_json::to_value(body)?))
    }
}

/// See the official [Delete Agent API reference](https://elevenlabs.io/docs/api-reference/delete-conversational-ai-agent)
///
/// # Example
///
/// ```no_run
/// use speakrs_elevenlabs::endpoints::convai::agents::DeleteAgent;
/// use speakrs_elevenlabs::{ElevenLabsClient, Result};
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///    let client = ElevenLabsClient::from_env()?;
///    let endpoint = DeleteAgent::new("agent_id");
///    let resp = client.hit(endpoint).await?;
///    println!("{:?}", resp);
///    Ok(())
/// }
/// ```
/// See [Delete Agent API reference](https://elevenlabs.io/docs/conversational-ai/api-reference/agents/delete-agent)
#[derive(Clone, Debug)]
pub struct DeleteAgent {
    agent_id: String,
}

impl DeleteAgent {
    pub fn new(agent_id: impl Into<String>) -> Self {
        DeleteAgent {
            agent_id: agent_id.into(),
        }
    }
}

impl ElevenLabsEndpoint for DeleteAgent {
    const PATH: &'static str = "/v1/convai/agents/:agent_id";

    const METHOD: Method = Method::DELETE;

    type ResponseBody = ();

    fn path_params(&self) -> Vec<(&'static str, &str)> {
        vec![self.agent_id.and_param(PathParam::AgentID)]
    }

    async fn response_body(self, _resp: Response) -> Result<Self::ResponseBody> {
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
/// Conversation configuration for an agent
pub struct ConversationConfig {
    /// Agent specific configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentConfig>,
    /// Configuration for conversational transcription
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asr: Option<ASR>,
    /// Configuration for conversational events
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<Conversation>,
    /// Configuration for conversational text to speech
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tts: Option<TTSConfig>,
    /// Configuration for turn detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn: Option<Turn>,
    /// Language presets for conversations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_presets: Option<HashMap<String, LanguagePreset>>,
}

impl ConversationConfig {
    pub fn with_agent_config(mut self, agent_config: AgentConfig) -> Self {
        self.agent = Some(agent_config);
        self
    }

    pub fn with_asr(mut self, asr: ASR) -> Self {
        self.asr = Some(asr);
        self
    }

    pub fn with_conversation(mut self, conversation: Conversation) -> Self {
        self.conversation = Some(conversation);
        self
    }

    pub fn with_tts_config(mut self, tts: TTSConfig) -> Self {
        self.tts = Some(tts);
        self
    }

    pub fn with_turn(mut self, turn: Turn) -> Self {
        self.turn = Some(turn);
        self
    }

    pub fn with_language_presets(
        mut self,
        language_presets: HashMap<String, LanguagePreset>,
    ) -> Self {
        self.language_presets = Some(language_presets);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AgentConfig {
    /// The prompt for the agent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptConfig>,
    /// If non-empty, the first message the agent will say.
    /// If empty, the agent waits for the user to start the discussion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_message: Option<String>,
    /// Language of the agent - used for ASR and TTS
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Configuration for dynamic variables
    ///
    /// See [Dynamic Variables](https://elevenlabs.io/docs/conversational-ai/customization/personalization/dynamic-variables)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_variables: Option<DynamicVariables>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DynamicVariables {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_variable_placeholders: Option<HashMap<String, DynamicVar>>,
}

impl DynamicVariables {
    pub fn new(placeholders: HashMap<String, DynamicVar>) -> Self {
        DynamicVariables {
            dynamic_variable_placeholders: Some(placeholders),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum DynamicVar {
    String(String),
    Int(i32),
    Double(f64),
    Bool(bool),
    Null(Value),
}

impl DynamicVar {
    pub fn new_string(value: impl Into<String>) -> Self {
        DynamicVar::String(value.into())
    }

    pub fn new_int(value: i32) -> Self {
        DynamicVar::Int(value)
    }

    pub fn new_double(value: f64) -> Self {
        DynamicVar::Double(value)
    }

    pub fn new_bool(value: bool) -> Self {
        DynamicVar::Bool(value)
    }
}

impl AgentConfig {
    pub fn new(
        prompt: PromptConfig,
        first_message: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        AgentConfig {
            prompt: Some(prompt),
            first_message: Some(first_message.into()),
            language: Some(language.into()),
            dynamic_variables: None,
        }
    }

    pub fn with_prompt(mut self, prompt: PromptConfig) -> Self {
        self.prompt = Some(prompt);
        self
    }

    pub fn with_first_message(mut self, first_message: impl Into<String>) -> Self {
        self.first_message = Some(first_message.into());
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    pub fn with_dynamic_variables(mut self, dynamic_variables: DynamicVariables) -> Self {
        self.dynamic_variables = Some(dynamic_variables);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PromptConfig {
    /// The prompt for the agent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// The LLM to query with the prompt and the chat history
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<LLM>,
    /// The temperature for the LLM
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// If greater than 0, maximum number of tokens the LLM can predict
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    /// A list of tools that the agent can use over the course of the conversation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// A list of IDs of tools used by the agent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_ids: Option<Vec<String>>,
    /// A list of knowledge bases to be used by the agent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knowledge_base: Option<Vec<KnowledgeBase>>,
    /// Definition for a custom LLM if LLM field is set to ‘CUSTOM_LLM’
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_llm: Option<CustomLLM>,
    /// Whether to ignore the default personality
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_default_personality: Option<bool>,
    /// Configuration for RAG
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rag: Option<RAG>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RAG {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<EmbeddingModel>,
    /// Maximum vector distance of retrieved chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_vector_distance: Option<f32>,
    /// Maximum total length of document chunks retrieved from RAG.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_documents_length: Option<u32>,
}

impl RAG {
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = Some(enabled);
        self
    }
    pub fn with_embedding_model(mut self, embedding_model: EmbeddingModel) -> Self {
        self.embedding_model = Some(embedding_model);
        self
    }

    pub fn with_max_vector_distance(mut self, max_vector_distance: f32) -> Self {
        self.max_vector_distance = Some(max_vector_distance);
        self
    }

    pub fn with_max_documents_length(mut self, max_documents_length: u32) -> Self {
        self.max_documents_length = Some(max_documents_length);
        self
    }
}

impl PromptConfig {
    pub fn with_knowledge_base(mut self, knowledge_base: Vec<KnowledgeBase>) -> Self {
        self.knowledge_base = Some(knowledge_base);
        self
    }

    pub fn with_llm(mut self, llm: LLM) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_custom_llm(mut self, custom_llm: CustomLLM) -> Self {
        self.custom_llm = Some(custom_llm);
        self
    }

    pub fn with_tool_ids(mut self, tool_ids: Vec<String>) -> Self {
        self.tool_ids = Some(tool_ids);
        self
    }

    pub fn ignore_default_personality(mut self, boolean: bool) -> Self {
        self.ignore_default_personality = Some(boolean);
        self
    }

    pub fn with_rag(mut self, rag: RAG) -> Self {
        self.rag = Some(rag);
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KnowledgeBase {
    pub id: String,
    pub name: String,
    pub usage_mode: Option<UsageMode>,
    pub r#type: KnowledgeBaseType,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum UsageMode {
    Prompt,
    Auto,
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeBaseType {
    File,
    Url,
}

impl KnowledgeBase {
    pub fn new_file(id: impl Into<String>, name: impl Into<String>) -> Self {
        KnowledgeBase {
            id: id.into(),
            name: name.into(),
            usage_mode: None,
            r#type: KnowledgeBaseType::File,
        }
    }

    pub fn new_url(id: impl Into<String>, name: impl Into<String>) -> Self {
        KnowledgeBase {
            id: id.into(),
            name: name.into(),
            usage_mode: None,
            r#type: KnowledgeBaseType::Url,
        }
    }

    pub fn with_usage_mode(mut self, usage_mode: UsageMode) -> Self {
        self.usage_mode = Some(usage_mode);
        self
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LLM {
    #[serde(rename = "gpt-4o-mini")]
    Gpt4oMini,
    #[serde(rename = "gpt-4o")]
    Gpt4o,
    #[serde(rename = "gpt-4")]
    Gpt4,
    #[serde(rename = "gpt-4-turbo")]
    Gpt4Turbo,
    #[serde(rename = "gpt-3.5-turbo")]
    Gpt3_5Turbo,
    #[serde(rename = "gemini-1.5-pro")]
    Gemini1_5Pro,
    #[serde(rename = "gemini-1.5-flash")]
    Gemini1_5Flash,
    #[serde(rename = "gemini-1.0-pro")]
    Gemini1_0Pro,
    #[serde(rename = "gemini-2.0-flash-001")]
    #[default]
    Gemini2_0Flash001,
    #[serde(rename = "gemini-2.0-flash-lite")]
    Gemini2_0FlashLite,
    #[serde(rename = "claude-3-5-sonnet")]
    Claude3_5Sonnet,
    #[serde(rename = "claude-3-7-sonnet")]
    Claude3_7Sonnet,
    #[serde(rename = "claude-3-5-sonnet-v1")]
    Claude3_5SonnetV1,
    #[serde(rename = "claude-3-haiku")]
    Claude3Haiku,
    #[serde(rename = "grok-beta")]
    GrokBeta,
    #[serde(rename = "custom-llm")]
    CustomLLM,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tool {
    r#type: ToolType,
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_schema: Option<ApiSchema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expects_response: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<ClientToolParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_timeout_secs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dynamic_variables: Option<DynamicVariables>,
}

impl Tool {
    pub fn new_webhook(webhook: WebHook) -> Self {
        Tool {
            r#type: ToolType::Webhook,
            name: webhook.name,
            description: webhook.description,
            api_schema: Some(webhook.api_schema),
            expects_response: None,
            parameters: None,
            response_timeout_secs: None,
            dynamic_variables: None,
        }
    }

    pub fn new_client(client_tool: ClientTool) -> Self {
        Tool {
            r#type: ToolType::Client,
            name: client_tool.name,
            description: client_tool.description,
            api_schema: None,
            expects_response: client_tool.expects_response,
            parameters: client_tool.parameters,
            response_timeout_secs: client_tool.response_timeout_secs,
            dynamic_variables: client_tool.dynamic_variables,
        }
    }

    pub fn new_system(system_tool: SystemTool) -> Self {
        Tool {
            r#type: ToolType::System,
            name: system_tool.name,
            description: system_tool.description.unwrap_or_default(),
            api_schema: None,
            expects_response: None,
            parameters: None,
            response_timeout_secs: None,
            dynamic_variables: None,
        }
    }
}

/// A webhook tool is a tool that calls an external webhook from ElevenLabs' server
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WebHook {
    api_schema: ApiSchema,
    description: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_variables: Option<DynamicVariables>,
    r#type: ToolType,
}

impl WebHook {
    pub fn new<T: Into<String>>(name: T, description: T, api_schema: ApiSchema) -> Self {
        WebHook {
            api_schema,
            description: description.into(),
            name: name.into(),
            r#type: ToolType::Webhook,
            dynamic_variables: None,
        }
    }
}

impl From<WebHook> for Tool {
    fn from(webhook: WebHook) -> Self {
        Tool::new_webhook(webhook)
    }
}

impl From<ClientTool> for Tool {
    fn from(client_tool: ClientTool) -> Self {
        Tool::new_client(client_tool)
    }
}

/// A client tool is one that sends an event to the user’s client to trigger something client side
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ClientTool {
    pub description: String,
    pub name: String,
    pub expects_response: Option<bool>,
    pub parameters: Option<ClientToolParams>,
    pub response_timeout_secs: Option<u32>,
    pub dynamic_variables: Option<DynamicVariables>,
    r#type: ToolType,
}

impl ClientTool {
    pub fn new<T: Into<String>>(name: T, description: T) -> Self {
        ClientTool {
            description: description.into(),
            name: name.into(),
            expects_response: None,
            parameters: None,
            response_timeout_secs: None,
            dynamic_variables: None,
            r#type: ToolType::Client,
        }
    }

    pub fn with_expects_response(mut self, expects_response: bool) -> Self {
        self.expects_response = Some(expects_response);
        self
    }

    pub fn with_parameters(mut self, parameters: ClientToolParams) -> Self {
        self.parameters = Some(parameters);
        self
    }

    pub fn with_response_timeout_secs(mut self, response_timeout_secs: u32) -> Self {
        self.response_timeout_secs = Some(response_timeout_secs);
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ClientToolParams {
    r#type: DataType,
    pub properties: Option<HashMap<String, Schema>>,
    pub required: Option<Vec<String>>,
    pub description: Option<String>,
}

impl ClientToolParams {
    pub fn with_properties(mut self, properties: HashMap<String, Schema>) -> Self {
        self.properties = Some(properties);
        self
    }

    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = Some(required);
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl Default for ClientToolParams {
    fn default() -> Self {
        ClientToolParams {
            r#type: DataType::Object,
            properties: None,
            required: None,
            description: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Schema {
    Literal(LiteralJsonSchema),
    Object(ObjectJsonSchema),
    Array(ArrayJsonSchema),
}

impl Schema {
    pub fn new_boolean(description: impl Into<String>) -> Self {
        Schema::Literal(LiteralJsonSchema {
            r#type: DataType::Boolean,
            description: description.into(),
            dynamic_variable: None,
        })
    }

    pub fn new_integer(description: impl Into<String>) -> Self {
        Schema::Literal(LiteralJsonSchema {
            r#type: DataType::Integer,
            description: description.into(),
            dynamic_variable: None,
        })
    }

    pub fn new_number(description: impl Into<String>) -> Self {
        Schema::Literal(LiteralJsonSchema {
            r#type: DataType::Number,
            description: description.into(),
            dynamic_variable: None,
        })
    }

    pub fn new_string(description: impl Into<String>) -> Self {
        Schema::Literal(LiteralJsonSchema {
            r#type: DataType::String,
            description: description.into(),
            dynamic_variable: None,
        })
    }

    pub fn new_object(properties: HashMap<String, Schema>) -> Self {
        Schema::Object(ObjectJsonSchema {
            r#type: DataType::Object,
            properties: Some(properties),
            required: None,
            description: None,
        })
    }

    pub fn new_array(items: Schema) -> Self {
        Schema::Array(ArrayJsonSchema {
            r#type: DataType::Array,
            items: Box::new(items),
            description: None,
        })
    }

    pub fn with_required(mut self, required: Vec<String>) -> Self {
        if let Schema::Object(obj) = &mut self {
            obj.required = Some(required);
        }
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        match &mut self {
            Schema::Literal(literal) => {
                literal.description = description.into();
            }
            Schema::Object(obj) => {
                obj.description = Some(description.into());
            }
            Schema::Array(array) => {
                array.description = Some(description.into());
            }
        }
        self
    }

    pub fn with_properties(mut self, properties: HashMap<String, Schema>) -> Self {
        if let Schema::Object(obj) = &mut self {
            obj.properties = Some(properties);
        }
        self
    }

    pub fn with_items(mut self, items: Schema) -> Self {
        if let Schema::Array(array) = &mut self {
            array.items = Box::new(items);
        }
        self
    }

    pub fn with_dynamic_variable(mut self, dynamic_variable: impl Into<String>) -> Self {
        if let Schema::Literal(literal) = &mut self {
            literal.dynamic_variable = Some(dynamic_variable.into());
        }
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LiteralJsonSchema {
    pub r#type: DataType,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_variable: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ObjectJsonSchema {
    r#type: DataType,
    pub properties: Option<HashMap<String, Schema>>,
    pub required: Option<Vec<String>>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArrayJsonSchema {
    r#type: DataType,
    items: Box<Schema>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CustomLLM {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<CustomAPIKey>,
}

impl CustomLLM {
    pub fn new(url: impl Into<String>) -> Self {
        CustomLLM {
            url: url.into(),
            model_id: None,
            api_key: None,
        }
    }

    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    pub fn with_apikey(mut self, apikey: CustomAPIKey) -> Self {
        self.api_key = Some(apikey);
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CustomAPIKey {
    pub secret_id: String,
}

/// Configuration for a webhook that will be called by an LLM tool.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ApiSchema {
    url: String,
    method: ApiMethod,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_params_schema: Option<HashMap<String, ParamSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    query_params_schema: Option<QueryParamsSchema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_body_schema: Option<RequestBodySchema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_headers: Option<HashMap<String, RequestHeaders>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum RequestHeaders {
    SecretLocator { secret_id: String },
    Value(String),
}

impl RequestHeaders {
    pub fn new(secret_id: impl Into<String>) -> Self {
        RequestHeaders::SecretLocator {
            secret_id: secret_id.into(),
        }
    }

    pub fn new_value(value: impl Into<String>) -> Self {
        RequestHeaders::Value(value.into())
    }
}

impl ApiSchema {
    pub fn new(url: &str) -> Self {
        ApiSchema {
            url: url.to_string(),
            ..Default::default()
        }
    }

    pub fn with_method(mut self, method: ApiMethod) -> Self {
        self.method = method;
        self
    }
    pub fn with_path_params(mut self, path_params_schema: HashMap<String, ParamSchema>) -> Self {
        self.path_params_schema = Some(path_params_schema);
        self
    }
    pub fn with_query_params(mut self, query_params_schema: QueryParamsSchema) -> Self {
        self.query_params_schema = Some(query_params_schema);
        self
    }

    pub fn with_request_body(mut self, request_body_schema: RequestBodySchema) -> Self {
        self.request_body_schema = Some(request_body_schema);
        self
    }

    pub fn with_request_headers(
        mut self,
        request_headers: HashMap<String, RequestHeaders>,
    ) -> Self {
        self.request_headers = Some(request_headers);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub enum ApiMethod {
    #[default]
    GET,
    POST,
    PATCH,
    DELETE,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParamSchema {
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    dynamic_variable: Option<String>,
    r#type: DataType,
}

impl ParamSchema {
    pub fn new_bool(description: impl Into<String>) -> Self {
        ParamSchema {
            description: description.into(),
            dynamic_variable: None,
            r#type: DataType::Boolean,
        }
    }

    pub fn new_integer(description: impl Into<String>) -> Self {
        ParamSchema {
            description: description.into(),
            dynamic_variable: None,
            r#type: DataType::Integer,
        }
    }

    pub fn new_number(description: impl Into<String>) -> Self {
        ParamSchema {
            description: description.into(),
            dynamic_variable: None,
            r#type: DataType::Number,
        }
    }

    pub fn new_string(description: impl Into<String>) -> Self {
        ParamSchema {
            description: description.into(),
            dynamic_variable: None,
            r#type: DataType::String,
        }
    }

    pub fn with_dynamic_variable(mut self, dynamic_variable: impl Into<String>) -> Self {
        self.dynamic_variable = Some(dynamic_variable.into());
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
    Double,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QueryParamsSchema {
    properties: HashMap<String, ParamSchema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    required: Option<Vec<String>>,
}

impl QueryParamsSchema {
    pub fn new(properties: HashMap<String, ParamSchema>) -> Self {
        QueryParamsSchema {
            properties,
            required: None,
        }
    }

    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = Some(required);
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RequestBodySchema {
    r#type: DataType,
    properties: HashMap<String, Schema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

impl RequestBodySchema {
    pub fn new(properties: HashMap<String, Schema>) -> Self {
        RequestBodySchema {
            r#type: DataType::Object,
            properties,
            required: None,
            description: None,
        }
    }

    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = Some(required);
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SystemTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    name: String,
    r#type: ToolType,
}

impl SystemTool {
    pub fn end_call() -> Self {
        SystemTool {
            description: None,
            name: "end_call".to_string(),
            r#type: ToolType::System,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Webhook,
    Client,
    System,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ASR {
    /// The quality of the transcription.
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<AsrQuality>,
    /// The provider of the transcription service
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<AsrProvider>,
    /// The format of the audio to be transcribed
    #[serde(skip_serializing_if = "Option::is_none")]
    user_input_audio_format: Option<ConvAIAudioFormat>,
    /// Keywords to boost prediction probability for
    #[serde(skip_serializing_if = "Vec::is_empty")]
    keywords: Vec<String>,
}

// impl `with_quality` and `with_provider` methods
// when enums have more than one variant
impl ASR {
    //pub fn with_quality(mut self, quality: AsrQuality) -> Self {
    //    self.quality = quality;
    //    self
    //}

    //pub fn with_provider(mut self, provider: AsrProvider) -> Self {
    //    self.provider = provider;
    //    self
    //}
    pub fn with_user_input_audio_format(
        mut self,
        user_input_audio_format: ConvAIAudioFormat,
    ) -> Self {
        self.user_input_audio_format = Some(user_input_audio_format);
        self
    }

    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AsrQuality {
    #[default]
    High,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AsrProvider {
    #[default]
    ElevenLabs,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub enum ConvAIAudioFormat {
    #[serde(rename = "pcm_8000")]
    Pcm8000hz,
    #[default]
    #[serde(rename = "pcm_16000")]
    Pcm16000hz,
    #[serde(rename = "pcm_22050")]
    Pcm22050hz,
    #[serde(rename = "pcm_24000")]
    Pcm24000hz,
    #[serde(rename = "pcm_44100")]
    Pcm44100hz,
    #[serde(rename = "ulaw_8000")]
    Ulaw8000hz,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Conversation {
    /// The events that will be sent to the client
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub client_events: Vec<ClientEvent>,
    /// The maximum duration of a conversation in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_duration_seconds: Option<u32>,
}

impl Conversation {
    pub fn with_client_events(mut self, client_events: Vec<ClientEvent>) -> Self {
        self.client_events = client_events;
        self
    }

    pub fn with_max_duration_seconds(mut self, max_duration_seconds: u32) -> Self {
        self.max_duration_seconds = Some(max_duration_seconds);
        self
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Conversation {
            client_events: vec![ClientEvent::Audio, ClientEvent::Interruption],
            max_duration_seconds: None,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TTSConfig {
    /// The model to use for TTS
    ///
    /// Default: `ConvAIModel::ElevenTurboV2`
    ///
    /// #### Additional Variants
    /// - `ConvAIModel::ElevenTurboV2_5`
    /// - `ConvAIModel::ElevenFlashV2`
    /// - `ConvAIModel::ElevenFlashV2_5`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<ConvAIModel>,
    /// The voice ID to use for TTS
    ///
    /// Default: `DefaultVoice::Eric` i.e. `cjVigY5qzO86Huf0OWal`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
    /// The audio format to use for TTS
    ///
    /// Default: `ConvAIAudioFormat::Pcm16000hz`
    ///
    /// #### Additional Variants
    /// - `ConvAIAudioFormat::Pcm8000hz`
    /// - `ConvAIAudioFormat::Pcm22050hz`
    /// - `ConvAIAudioFormat::Pcm24000hz`
    /// - `ConvAIAudioFormat::Pcm44100hz`
    /// - `ConvAIAudioFormat::Ulaw8000hz`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_output_audio_format: Option<ConvAIAudioFormat>,
    /// The optimization for streaming latency
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimize_streaming_latency: Option<u32>,
    /// The stability of generated speech
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// The speed of generated speech
    pub speed: Option<f32>,
    /// The similarity boost for generated speech
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_boost: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    /// The pronunciation dictionary locators
    pub pronunciation_dictionary_locators: Vec<DictionaryLocator>,
}

impl TTSConfig {
    pub fn with_model_id(mut self, model_id: ConvAIModel) -> Self {
        self.model_id = Some(model_id);
        self
    }

    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = Some(voice_id.into());
        self
    }

    pub fn with_agent_output_audio_format(
        mut self,
        agent_output_audio_format: ConvAIAudioFormat,
    ) -> Self {
        self.agent_output_audio_format = Some(agent_output_audio_format);
        self
    }

    pub fn with_optimize_streaming_latency(mut self, optimize_streaming_latency: u32) -> Self {
        self.optimize_streaming_latency = Some(optimize_streaming_latency);
        self
    }

    pub fn with_stability(mut self, stability: f32) -> Self {
        self.stability = Some(stability);
        self
    }

    pub fn with_similarity_boost(mut self, similarity_boost: f32) -> Self {
        self.similarity_boost = Some(similarity_boost);
        self
    }

    pub fn with_pronunciation_dictionary_locators(
        mut self,
        pronunciation_dictionary_locators: Vec<DictionaryLocator>,
    ) -> Self {
        self.pronunciation_dictionary_locators = pronunciation_dictionary_locators;
        self
    }

    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientEvent {
    AgentResponse,
    AgentResponseCorrection,
    AsrInitiationMetadata,
    Audio,
    ClientToolCall,
    ConversationInitiationMetadata,
    InternalTentativeAgentResponse,
    InternalTurnProbability,
    InternalVadScore,
    Interruption,
    Ping,
    UserTranscript,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub enum ConvAIModel {
    #[default]
    #[serde(rename = "eleven_turbo_v2")]
    ElevenTurboV2,
    #[serde(rename = "eleven_turbo_v2_5")]
    ElevenTurboV2_5,
    #[serde(rename = "eleven_flash_v2")]
    ElevenFlashV2,
    #[serde(rename = "eleven_flash_v2_5")]
    ElevenFlashV2_5,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Turn {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Maximum wait time for the user’s reply before re-engaging the user
    pub turn_timeout: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// The mode of turn detection
    pub mode: Option<TurnMode>,
}

impl Turn {
    pub fn with_mode(mut self, mode: TurnMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn with_turn_timeout(mut self, turn_timeout: f32) -> Self {
        self.turn_timeout = Some(turn_timeout);
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TurnMode {
    Silence,
    Turn,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct LanguagePreset {
    pub overrides: ConversationConfigOverride,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_message_translation: Option<FirstMessageTranslation>,
}

impl LanguagePreset {
    pub fn new(overrides: ConversationConfigOverride) -> Self {
        LanguagePreset {
            overrides,
            first_message_translation: None,
        }
    }

    pub fn with_first_message_translation(
        mut self,
        first_message_translation: FirstMessageTranslation,
    ) -> Self {
        self.first_message_translation = Some(first_message_translation);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct FirstMessageTranslation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PlatformSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<Auth>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<Evaluation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub widget: Option<Widget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overrides: Option<Overrides>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ban: Option<Ban>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety: Option<Safety>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub privacy: Option<Privacy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_limits: Option<CallLimits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_overrides: Option<WorkspaceOverrides>,
}

impl PlatformSettings {
    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = Some(auth);
        self
    }

    pub fn with_evaluation(mut self, evaluation: Evaluation) -> Self {
        self.evaluation = Some(evaluation);
        self
    }

    pub fn with_widget(mut self, widget: Widget) -> Self {
        self.widget = Some(widget);
        self
    }

    pub fn with_data_collection(mut self, data_collection: DataCollection) -> Self {
        self.data_collection = Some(data_collection);
        self
    }

    pub fn with_overrides(mut self, overrides: Overrides) -> Self {
        self.overrides = Some(overrides);
        self
    }

    pub fn with_ban(mut self, ban: Ban) -> Self {
        self.ban = Some(ban);
        self
    }

    pub fn with_safety(mut self, safety: Safety) -> Self {
        self.safety = Some(safety);
        self
    }

    pub fn with_privacy(mut self, privacy: Privacy) -> Self {
        self.privacy = Some(privacy);
        self
    }

    pub fn with_call_limits(mut self, call_limits: CallLimits) -> Self {
        self.call_limits = Some(call_limits);
        self
    }

    pub fn with_workspace_overrides(mut self, workspace_overrides: WorkspaceOverrides) -> Self {
        self.workspace_overrides = Some(workspace_overrides);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Auth {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_auth: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowlist: Option<Vec<AllowHost>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shareable_token: Option<String>,
}

impl Auth {
    pub fn with_enable_auth(mut self, enable_auth: bool) -> Self {
        self.enable_auth = Some(enable_auth);
        self
    }

    pub fn with_allowlist<'a, I: IntoIterator<Item = &'a str>>(mut self, allowlist: I) -> Self {
        let allowlist = allowlist.into_iter().map(AllowHost::new).collect();

        self.allowlist = Some(allowlist);
        self
    }

    pub fn with_shareable_token(mut self, shareable_token: impl Into<String>) -> Self {
        self.shareable_token = Some(shareable_token.into());
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AllowHost {
    hostname: String,
}

impl AllowHost {
    fn new(hostname: &str) -> Self {
        AllowHost {
            hostname: hostname.to_string(),
        }
    }
}

pub type DataCollection = HashMap<String, CustomData>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CustomData {
    pub description: String,
    pub r#type: DataType,
    pub dynamic_variable: Option<String>,
    pub constant: Option<DataType>,
}

impl CustomData {
    pub fn new_boolean(description: impl Into<String>) -> Self {
        CustomData {
            description: description.into(),
            r#type: DataType::Boolean,
            dynamic_variable: None,
            constant: None,
        }
    }

    pub fn new_integer(description: impl Into<String>) -> Self {
        CustomData {
            description: description.into(),
            r#type: DataType::Integer,
            dynamic_variable: None,
            constant: None,
        }
    }

    pub fn new_number(description: impl Into<String>) -> Self {
        CustomData {
            description: description.into(),
            r#type: DataType::Number,
            dynamic_variable: None,
            constant: None,
        }
    }

    pub fn new_string(description: impl Into<String>) -> Self {
        CustomData {
            description: description.into(),
            r#type: DataType::String,
            dynamic_variable: None,
            constant: None,
        }
    }

    pub fn with_dynamic_variable(mut self, dynamic_variable: impl Into<String>) -> Self {
        self.dynamic_variable = Some(dynamic_variable.into());
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Evaluation {
    pub criteria: Vec<Criterion>,
}

impl Evaluation {
    pub fn new(criteria: Vec<Criterion>) -> Self {
        Evaluation { criteria }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Criterion {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    r#type: CriterionType,
    pub conversation_goal_prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_knowledge_base: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum CriterionType {
    #[default]
    Prompt,
}

impl Criterion {
    pub fn new(id: impl Into<String>, conversation_goal_prompt: impl Into<String>) -> Self {
        Criterion {
            id: id.into(),
            name: None,
            r#type: CriterionType::Prompt,
            conversation_goal_prompt: conversation_goal_prompt.into(),
            use_knowledge_base: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_use_knowledge_base(mut self, use_knowledge_base: bool) -> Self {
        self.use_knowledge_base = Some(use_knowledge_base);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Overrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_config_override: Option<ConversationConfigOverride>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_llm_extra_body: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_conversation_initiation_client_data_from_webhook: Option<bool>,
}

impl Overrides {
    pub fn with_conversation_config_override(
        mut self,
        conversation_config_override: ConversationConfigOverride,
    ) -> Self {
        self.conversation_config_override = Some(conversation_config_override);
        self
    }

    pub fn override_custom_llm_extra_body(mut self, custom_llm_extra_body: bool) -> Self {
        self.custom_llm_extra_body = Some(custom_llm_extra_body);
        self
    }

    pub fn enable_conversation_initiation_client_data_from_webhook(
        mut self,
        boolean: bool,
    ) -> Self {
        self.enable_conversation_initiation_client_data_from_webhook = Some(boolean);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ConversationConfigOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentOverride>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tts: Option<TTSOverride>,
}

impl ConversationConfigOverride {
    pub fn with_agent_override(mut self, agent: AgentOverride) -> Self {
        self.agent = Some(agent);
        self
    }

    pub fn with_tts_override(mut self, tts: TTSOverride) -> Self {
        self.tts = Some(tts);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AgentOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptOverride>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_message: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<bool>,
}

impl AgentOverride {
    pub fn with_prompt_override(mut self, prompt: PromptOverride) -> Self {
        self.prompt = Some(prompt);
        self
    }

    pub fn override_first_message(mut self, first_message: bool) -> Self {
        self.first_message = Some(first_message);
        self
    }

    pub fn override_language(mut self, language: bool) -> Self {
        self.language = Some(language);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PromptOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<bool>,
}

impl PromptOverride {
    pub fn override_prompt(mut self, prompt: bool) -> Self {
        self.prompt = Some(prompt);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TTSOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<bool>,
}

impl TTSOverride {
    pub fn override_voice_id(mut self, voice_id: bool) -> Self {
        self.voice_id = Some(voice_id);
        self
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Ban {
    pub at_unix: u64,
    pub reason_type: BanReasonType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BanReasonType {
    Safety,
    Manual,
}

impl Ban {
    pub fn new_safety(at_unix: u64) -> Self {
        Ban {
            at_unix,
            reason_type: BanReasonType::Safety,
            reason: None,
        }
    }

    pub fn new_manual(at_unix: u64) -> Self {
        Ban {
            at_unix,
            reason_type: BanReasonType::Manual,
            reason: None,
        }
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Safety {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_blocked_ivc: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_blocked_non_ivc: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Privacy {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_voice: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retention_days: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delete_transcript_and_pii: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delete_audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apply_to_existing_conversations: Option<bool>,
}

impl Privacy {
    pub fn record_voice(mut self, record_voice: bool) -> Self {
        self.record_voice = Some(record_voice);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CallLimits {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_concurrency_limit: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub daily_limit: Option<u32>,
}

impl CallLimits {
    pub fn with_agent_concurrency_limit(mut self, agent_concurrency_limit: i32) -> Self {
        self.agent_concurrency_limit = Some(agent_concurrency_limit);
        self
    }

    pub fn with_daily_limit(mut self, daily_limit: u32) -> Self {
        self.daily_limit = Some(daily_limit);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct WorkspaceOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_initiation_client_data_webhook:
        Option<ConversationInitiationClientDataWebhook>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhooks: Option<Webhooks>,
}

impl WorkspaceOverrides {
    pub fn with_conversation_initiation_client_data_webhook(
        mut self,
        webhook: ConversationInitiationClientDataWebhook,
    ) -> Self {
        self.conversation_initiation_client_data_webhook = Some(webhook);
        self
    }

    pub fn with_webhooks(mut self, webhooks: Webhooks) -> Self {
        self.webhooks = Some(webhooks);
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Widget {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<WidgetVariant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expandable: Option<Expandable>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avatar: Option<Avatar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback_mode: Option<FeedBackMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_avatar_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bg_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub btn_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub btn_text_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub border_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub border_radius: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub btn_radius: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_call_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_call_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expand_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub listening_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaking_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shareable_page_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_page_show_terms: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terms_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terms_html: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terms_keys: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_avatar_when_collapsed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_banner: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mic_muting_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_selector: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WidgetVariant {
    Compact,
    Full,
    Expandable,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Expandable {
    Never,
    Mobile,
    Desktop,
    Always,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Avatar {
    Image {
        r#type: AvatarType,
        url: Option<String>,
    },
    Orb {
        r#type: AvatarType,
        color_1: Option<String>,
        color_2: Option<String>,
    },
    Url {
        r#type: AvatarType,
        custom_url: Option<String>,
    },
}

impl Avatar {
    pub fn default_image() -> Self {
        Avatar::Image {
            r#type: AvatarType::Image,
            url: None,
        }
    }

    pub fn default_orb() -> Self {
        Avatar::Orb {
            r#type: AvatarType::Orb,
            color_1: None,
            color_2: None,
        }
    }

    pub fn default_url() -> Self {
        Avatar::Url {
            r#type: AvatarType::Url,
            custom_url: None,
        }
    }

    pub fn with_custom_url(mut self, custom_url: &str) -> Self {
        if let Avatar::Image { ref mut url, .. } = self {
            *url = Some(custom_url.to_string());
        }
        self
    }

    pub fn with_color_1(mut self, color: &str) -> Self {
        if let Avatar::Orb {
            ref mut color_1, ..
        } = self
        {
            *color_1 = Some(color.to_string());
        }
        self
    }

    pub fn with_color_2(mut self, color: &str) -> Self {
        if let Avatar::Orb {
            ref mut color_2, ..
        } = self
        {
            *color_2 = Some(color.to_string());
        }
        self
    }

    pub fn with_url(mut self, url: &str) -> Self {
        if let Avatar::Url {
            ref mut custom_url, ..
        } = self
        {
            *custom_url = Some(url.to_string());
        }
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AvatarType {
    Image,
    #[default]
    Orb,
    Url,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FeedBackMode {
    None,
    During,
    End,
}

impl Widget {
    pub fn with_variant(mut self, variant: WidgetVariant) -> Self {
        self.variant = Some(variant);
        self
    }

    pub fn with_avatar(mut self, avatar: Avatar) -> Self {
        self.avatar = Some(avatar);
        self
    }

    pub fn with_feedback_mode(mut self, feedback_mode: FeedBackMode) -> Self {
        self.feedback_mode = Some(feedback_mode);
        self
    }

    pub fn with_custom_avatar_path(mut self, custom_avatar_path: impl Into<String>) -> Self {
        self.custom_avatar_path = Some(custom_avatar_path.into());
        self
    }

    pub fn with_bg_color(mut self, bg_color: impl Into<String>) -> Self {
        self.bg_color = Some(bg_color.into());
        self
    }

    pub fn with_text_color(mut self, text_color: impl Into<String>) -> Self {
        self.text_color = Some(text_color.into());
        self
    }

    pub fn with_btn_color(mut self, btn_color: impl Into<String>) -> Self {
        self.btn_color = Some(btn_color.into());
        self
    }

    pub fn with_btn_text_color(mut self, btn_text_color: impl Into<String>) -> Self {
        self.btn_text_color = Some(btn_text_color.into());
        self
    }

    pub fn with_border_color(mut self, border_color: impl Into<String>) -> Self {
        self.border_color = Some(border_color.into());
        self
    }

    pub fn with_focus_color(mut self, focus_color: impl Into<String>) -> Self {
        self.focus_color = Some(focus_color.into());
        self
    }

    pub fn with_border_radius(mut self, border_radius: i64) -> Self {
        self.border_radius = Some(border_radius);
        self
    }

    pub fn with_btn_radius(mut self, btn_radius: i64) -> Self {
        self.btn_radius = Some(btn_radius);
        self
    }

    pub fn with_action_text(mut self, action_text: impl Into<String>) -> Self {
        self.action_text = Some(action_text.into());
        self
    }

    pub fn with_start_call_text(mut self, start_call_text: impl Into<String>) -> Self {
        self.start_call_text = Some(start_call_text.into());
        self
    }

    pub fn with_end_call_text(mut self, end_call_text: impl Into<String>) -> Self {
        self.end_call_text = Some(end_call_text.into());
        self
    }

    pub fn with_expand_text(mut self, expand_text: impl Into<String>) -> Self {
        self.expand_text = Some(expand_text.into());
        self
    }

    pub fn with_listening_text(mut self, listening_text: impl Into<String>) -> Self {
        self.listening_text = Some(listening_text.into());
        self
    }

    pub fn with_speaking_text(mut self, speaking_text: impl Into<String>) -> Self {
        self.speaking_text = Some(speaking_text.into());
        self
    }

    pub fn with_shareable_page_text(mut self, shareable_page_text: impl Into<String>) -> Self {
        self.shareable_page_text = Some(shareable_page_text.into());
        self
    }
}

/// Retrieve config for an agent
///
/// # Example
///
/// ```no_run
/// use speakrs_elevenlabs::endpoints::convai::agents::GetAgent;
/// use speakrs_elevenlabs::{ElevenLabsClient, Result};
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///    let client = ElevenLabsClient::from_env()?;
///    let resp = client.hit(GetAgent::new("agent_id")).await?;
///    println!("{:?}", resp);
/// Ok(())
/// }
/// ```
///
/// See [Get Agent API reference](https://elevenlabs.io/docs/conversational-ai/api-reference/agents/get-agent)
#[derive(Clone, Debug, Serialize)]
pub struct GetAgent {
    agent_id: String,
}

impl GetAgent {
    pub fn new(agent_id: impl Into<String>) -> Self {
        GetAgent {
            agent_id: agent_id.into(),
        }
    }
}

impl ElevenLabsEndpoint for GetAgent {
    const PATH: &'static str = "/v1/convai/agents/:agent_id";

    const METHOD: Method = Method::GET;

    type ResponseBody = GetAgentResponse;

    fn path_params(&self) -> Vec<(&'static str, &str)> {
        vec![self.agent_id.and_param(PathParam::AgentID)]
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.json().await?)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct GetAgentResponse {
    pub agent_id: String,
    pub name: String,
    pub conversation_config: ConversationConfig,
    pub platform_settings: Option<PlatformSettings>,
    pub metadata: Metadata,
    pub phone_numbers: Option<Vec<PhoneNumber>>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Metadata {
    pub created_at_unix_secs: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PhoneNumber {
    pub phone_number: String,
    pub provider: PhoneNumberProvider,
    pub label: String,
    pub phone_number_id: String,
    pub assigned_agent: AssignedAgent,
}

/// Returns a page of your agents and their metadata.
///
///
///
/// # Query Parameters
///
/// - `search` (optional): A search term to filter agents by name.
/// - `page_size` (optional): The number of agents to return per page. Can not exceed 100, default is 30.
/// - `cursor` (optional): A cursor to paginate through the list of agents.
///
/// # Response
///
/// The response will contain a list of agents and metadata about the list.
///
/// - `agents`: A `Vec<Agent>`.
/// - `has_more`: A boolean indicating if there are more agents to retrieve.
/// - `next_cursor`: A cursor to paginate to the next page of agents.
///
/// # Example
///
/// ```no_run
/// use speakrs_elevenlabs::endpoints::convai::agents::{GetAgents, GetAgentsQuery};
/// use speakrs_elevenlabs::{ElevenLabsClient, Result};
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///    let client = ElevenLabsClient::from_env()?;
///    let query = GetAgentsQuery::default().with_page_size(3);
///    let agents = client.hit(GetAgents::with_query(query)).await?;
///    for agent in agents {
///         println!("{:?}", agent);
///   }
///   Ok(())
/// }
/// ```
/// See [Get Agents API reference](https://elevenlabs.io/docs/conversational-ai/api-reference/agents/get-agents)
#[derive(Clone, Debug, Default, Serialize)]
pub struct GetAgents {
    query: Option<GetAgentsQuery>,
}

impl GetAgents {
    pub fn with_query(query: GetAgentsQuery) -> Self {
        GetAgents { query: Some(query) }
    }
}

impl ElevenLabsEndpoint for GetAgents {
    const PATH: &'static str = "/v1/convai/agents";

    const METHOD: Method = Method::GET;

    type ResponseBody = GetAgentsResponse;

    fn query_params(&self) -> Option<QueryValues> {
        self.query.as_ref().map(|q| q.params.clone())
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.json().await?)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct GetAgentsResponse {
    pub agents: Vec<Agent>,
    pub has_more: bool,
    pub next_cursor: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Agent {
    pub agent_id: String,
    pub name: String,
    pub created_at_unix_secs: u64,
    pub access_info: AccessInfo,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AccessInfo {
    pub is_creator: bool,
    pub creator_name: String,
    pub creator_email: String,
    pub role: AccessLevel,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GetAgentsQuery {
    params: QueryValues,
}

impl GetAgentsQuery {
    pub fn with_search(mut self, search: impl Into<String>) -> Self {
        self.params.push(("search", search.into()));
        self
    }

    pub fn with_page_size(mut self, page_size: u32) -> Self {
        self.params.push(("page_size", page_size.to_string()));
        self
    }

    pub fn with_cursor(mut self, cursor: impl Into<String>) -> Self {
        self.params.push(("cursor", cursor.into()));
        self
    }
}
/// Patches an Agent settings
///
/// # Example
///
/// ```no_run
/// use speakrs_elevenlabs::endpoints::convai::agents::*;
/// use speakrs_elevenlabs::{DefaultVoice, ElevenLabsClient, Result};
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let client = ElevenLabsClient::from_env()?;
///
///
///     let updated_config = ConversationConfig::default()
///         .with_agent_config(AgentConfig::default().with_first_message("updated first message"))
///         .with_tts_config(TTSConfig::default().with_voice_id(DefaultVoice::Matilda))
///         .with_conversation(Conversation::default().with_max_duration_seconds(60));
///
///     let body = UpdateAgentBody::default()
///         .with_conversation_config(updated_config)
///         .with_name("updated agent");
///
///     let endpoint = UpdateAgent::new("agent_id", body);
///
///     let resp = client.hit(endpoint).await?;
///
///     println!("{:?}", resp);
///
///     Ok(())
/// }
/// ```
/// See [Update Agent API reference](https://elevenlabs.io/docs/conversational-ai/api-reference/agents/update-agent)
#[derive(Clone, Debug)]
pub struct UpdateAgent {
    agent_id: String,
    body: UpdateAgentBody,
    query: Option<AgentQuery>,
}

impl UpdateAgent {
    pub fn new(agent_id: &str, body: UpdateAgentBody) -> Self {
        UpdateAgent {
            agent_id: agent_id.to_string(),
            body,
            query: None,
        }
    }

    pub fn with_query(mut self, query: AgentQuery) -> Self {
        self.query = Some(query);
        self
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AgentQuery {
    pub params: QueryValues,
}

impl AgentQuery {
    pub fn use_tool_ids(mut self) -> Self {
        self.params.push(("use_tool_ids", true.to_string()));
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct UpdateAgentBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    conversation_config: Option<ConversationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    platform_settings: Option<PlatformSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl UpdateAgentBody {
    pub fn with_conversation_config(mut self, conversation_config: ConversationConfig) -> Self {
        self.conversation_config = Some(conversation_config);
        self
    }
    pub fn with_platform_settings(mut self, platform_settings: PlatformSettings) -> Self {
        self.platform_settings = Some(platform_settings);
        self
    }
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

type UpdateAgentResponse = GetAgentResponse;

impl ElevenLabsEndpoint for UpdateAgent {
    const PATH: &'static str = "/v1/convai/agents/:agent_id";

    const METHOD: Method = Method::PATCH;

    type ResponseBody = UpdateAgentResponse;

    fn query_params(&self) -> Option<QueryValues> {
        self.query.as_ref().map(|q| q.params.clone())
    }

    fn path_params(&self) -> Vec<(&'static str, &str)> {
        vec![self.agent_id.and_param(PathParam::AgentID)]
    }

    async fn request_body(&self) -> Result<RequestBody> {
        TryInto::try_into(&self.body)
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.json().await?)
    }
}

impl TryInto<RequestBody> for &UpdateAgentBody {
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn try_into(self) -> Result<RequestBody> {
        Ok(RequestBody::Json(serde_json::to_value(self)?))
    }
}

/// Get the current link used to share the agent with others
///
/// # Example
///
/// ```no_run
/// use speakrs_elevenlabs::endpoints::convai::agents::GetLink;
/// use speakrs_elevenlabs::{ElevenLabsClient, Result};
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///    let client = ElevenLabsClient::from_env()?;
///    let resp = client.hit(GetLink::new("agent_id")).await?;
///    println!("{:?}", resp);
///    Ok(())
/// }
/// ```
/// See [Get Link API reference](https://elevenlabs.io/docs/conversational-ai/api-reference/agents/get-agent-link)
#[derive(Clone, Debug)]
pub struct GetLink {
    agent_id: String,
}

impl GetLink {
    pub fn new(agent_id: impl Into<String>) -> Self {
        GetLink {
            agent_id: agent_id.into(),
        }
    }
}

impl ElevenLabsEndpoint for GetLink {
    const PATH: &'static str = "/v1/convai/agents/:agent_id/link";

    const METHOD: Method = Method::GET;

    type ResponseBody = GetLinkResponse;

    fn path_params(&self) -> Vec<(&'static str, &str)> {
        vec![self.agent_id.and_param(PathParam::AgentID)]
    }

    async fn response_body(self, resp: Response) -> Result<Self::ResponseBody> {
        Ok(resp.json().await?)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct GetLinkResponse {
    pub agent_id: String,
    pub token: Option<Token>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Token {
    pub agent_id: String,
    pub conversation_token: String,
    pub expiration_time_unix_secs: Option<u64>,
    pub purpose: Option<Purpose>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    SignedUrl,
    ShareableLink,
}

impl IntoIterator for GetAgentsResponse {
    type Item = Agent;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.agents.into_iter()
    }
}

impl<'a> IntoIterator for &'a GetAgentsResponse {
    type Item = &'a Agent;
    type IntoIter = std::slice::Iter<'a, Agent>;

    fn into_iter(self) -> Self::IntoIter {
        self.agents.iter()
    }
}
