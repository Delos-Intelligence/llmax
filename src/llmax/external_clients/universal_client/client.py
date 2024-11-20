# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Any, Callable


from .chat import Chat
from .chat_completion import ChatCompletion


class UniversalClient:
    internal_client: Any
    chat: Chat

    def __init__(
        self,
        client_creation: Callable[..., Any],
        completion_call: Callable[..., ChatCompletion],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Construct a new synchronous client instance based on OpenAI client model."""
        self.internal_client = client_creation(*args, **kwargs)
        self.chat = Chat(self.internal_client, completion_call)


    # def __init__(
    #     self,
    #     *,
    #     api_key: str | None = None,
    #     project_id: str | None = None,
    #     endpoint: str | None = None,
    #     timeout: Optional[float] = None,
    #     max_retries: int = 3,
    # ) -> None:
    #     """Construct a new synchronous openai client instance.

    #     This automatically infers the following arguments from their corresponding environment variables if they are not provided:
    #     - `api_key` from `API_KEY`
    #     - `project` from `PROJECT_ID`
    #     """
    #     if api_key is None:
    #         api_key = os.environ.get("API_KEY")
    #     if api_key is None:
    #         raise Exception(
    #             "The api_key client option must be set either by passing api_key to the client or by setting the API_KEY environment variable"
    #         )
    #     self.api_key = api_key

    #     if project_id is None:
    #         project_id = os.environ.get("PROJECT_ID")
    #     self.project_id = project_id

    #     # self._default_stream_cls = Stream

    #     self.chat = Chat()

    # @property
    # @override
    # def auth_headers(self) -> dict[str, str]:
    #     api_key = self.api_key
    #     return {"Authorization": f"Bearer {api_key}"}


    # @override
    # def _make_status_error(
    #     self,
    #     err_msg: str,
    #     *,
    #     body: object,
    #     response: httpx.Response,
    # ) -> APIStatusError:
    #     data = body.get("error", body) if is_mapping(body) else body
    #     if response.status_code == 400:
    #         return _exceptions.BadRequestError(err_msg, response=response, body=data)

    #     if response.status_code == 401:
    #         return _exceptions.AuthenticationError(err_msg, response=response, body=data)

    #     if response.status_code == 403:
    #         return _exceptions.PermissionDeniedError(err_msg, response=response, body=data)

    #     if response.status_code == 404:
    #         return _exceptions.NotFoundError(err_msg, response=response, body=data)

    #     if response.status_code == 409:
    #         return _exceptions.ConflictError(err_msg, response=response, body=data)

    #     if response.status_code == 422:
    #         return _exceptions.UnprocessableEntityError(err_msg, response=response, body=data)

    #     if response.status_code == 429:
    #         return _exceptions.RateLimitError(err_msg, response=response, body=data)

    #     if response.status_code >= 500:
    #         return _exceptions.InternalServerError(err_msg, response=response, body=data)
    #     return APIStatusError(err_msg, response=response, body=data)


# class AsyncOpenAI(AsyncAPIClient):
#     completions: resources.AsyncCompletions
#     chat: resources.AsyncChat
#     embeddings: resources.AsyncEmbeddings
#     files: resources.AsyncFiles
#     images: resources.AsyncImages
#     audio: resources.AsyncAudio
#     models: resources.AsyncModels
#     with_raw_response: AsyncOpenAIWithRawResponse
#     with_streaming_response: AsyncOpenAIWithStreamedResponse

#     # client options
#     api_key: str
#     organization: str | None
#     project: str | None

#     def __init__(
#         self,
#         *,
#         api_key: str | None = None,
#         organization: str | None = None,
#         project: str | None = None,
#         base_url: str | httpx.URL | None = None,
#         timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
#         max_retries: int = DEFAULT_MAX_RETRIES,
#         default_headers: Mapping[str, str] | None = None,
#         default_query: Mapping[str, object] | None = None,
#         http_client: httpx.AsyncClient | None = None,
#     ) -> None:
#         """Construct a new async openai client instance.

#         This automatically infers the following arguments from their corresponding environment variables if they are not provided:
#         - `api_key` from `OPENAI_API_KEY`
#         - `organization` from `OPENAI_ORG_ID`
#         - `project` from `OPENAI_PROJECT_ID`
#         """
#         if api_key is None:
#             api_key = os.environ.get("OPENAI_API_KEY")
#         if api_key is None:
#             raise OpenAIError(
#                 "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
#             )
#         self.api_key = api_key

#         if organization is None:
#             organization = os.environ.get("OPENAI_ORG_ID")
#         self.organization = organization

#         if project is None:
#             project = os.environ.get("OPENAI_PROJECT_ID")
#         self.project = project

#         if base_url is None:
#             base_url = os.environ.get("OPENAI_BASE_URL")
#         if base_url is None:
#             base_url = f"https://api.openai.com/v1"


#         self._default_stream_cls = AsyncStream

#         self.completions = resources.AsyncCompletions(self)
#         self.chat = resources.AsyncChat(self)
#         self.embeddings = resources.AsyncEmbeddings(self)
#         self.files = resources.AsyncFiles(self)
#         self.images = resources.AsyncImages(self)
#         self.audio = resources.AsyncAudio(self)
#         self.moderations = resources.AsyncModerations(self)
#         self.models = resources.AsyncModels(self)
#         self.fine_tuning = resources.AsyncFineTuning(self)
#         self.beta = resources.AsyncBeta(self)
#         self.batches = resources.AsyncBatches(self)
#         self.uploads = resources.AsyncUploads(self)
#         self.with_raw_response = AsyncOpenAIWithRawResponse(self)
#         self.with_streaming_response = AsyncOpenAIWithStreamedResponse(self)

#     @property
#     @override
#     def qs(self) -> Querystring:
#         return Querystring(array_format="brackets")

#     @property
#     @override
#     def auth_headers(self) -> dict[str, str]:
#         api_key = self.api_key
#         return {"Authorization": f"Bearer {api_key}"}

#     @property
#     @override
#     def default_headers(self) -> dict[str, str | Omit]:
#         return {
#             **super().default_headers,
#             "X-Stainless-Async": f"async:{get_async_library()}",
#             "OpenAI-Organization": self.organization if self.organization is not None else Omit(),
#             "OpenAI-Project": self.project if self.project is not None else Omit(),
#             **self._custom_headers,
#         }

#     def copy(
#         self,
#         *,
#         api_key: str | None = None,
#         organization: str | None = None,
#         project: str | None = None,
#         base_url: str | httpx.URL | None = None,
#         timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
#         http_client: httpx.AsyncClient | None = None,
#         max_retries: int | NotGiven = NOT_GIVEN,
#         default_headers: Mapping[str, str] | None = None,
#         set_default_headers: Mapping[str, str] | None = None,
#         default_query: Mapping[str, object] | None = None,
#         set_default_query: Mapping[str, object] | None = None,
#         _extra_kwargs: Mapping[str, Any] = {},
#     ) -> Self:
#         """
#         Create a new client instance re-using the same options given to the current client with optional overriding.
#         """
#         if default_headers is not None and set_default_headers is not None:
#             raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

#         if default_query is not None and set_default_query is not None:
#             raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

#         headers = self._custom_headers
#         if default_headers is not None:
#             headers = {**headers, **default_headers}
#         elif set_default_headers is not None:
#             headers = set_default_headers

#         params = self._custom_query
#         if default_query is not None:
#             params = {**params, **default_query}
#         elif set_default_query is not None:
#             params = set_default_query

#         http_client = http_client or self._client
#         return self.__class__(
#             api_key=api_key or self.api_key,
#             organization=organization or self.organization,
#             project=project or self.project,
#             base_url=base_url or self.base_url,
#             timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
#             http_client=http_client,
#             max_retries=max_retries if is_given(max_retries) else self.max_retries,
#             default_headers=headers,
#             default_query=params,
#             **_extra_kwargs,
#         )

#     # Alias for `copy` for nicer inline usage, e.g.
#     # client.with_options(timeout=10).foo.create(...)
#     with_options = copy

#     @override
#     def _make_status_error(
#         self,
#         err_msg: str,
#         *,
#         body: object,
#         response: httpx.Response,
#     ) -> APIStatusError:
#         data = body.get("error", body) if is_mapping(body) else body
#         if response.status_code == 400:
#             return _exceptions.BadRequestError(err_msg, response=response, body=data)

#         if response.status_code == 401:
#             return _exceptions.AuthenticationError(err_msg, response=response, body=data)

#         if response.status_code == 403:
#             return _exceptions.PermissionDeniedError(err_msg, response=response, body=data)

#         if response.status_code == 404:
#             return _exceptions.NotFoundError(err_msg, response=response, body=data)

#         if response.status_code == 409:
#             return _exceptions.ConflictError(err_msg, response=response, body=data)

#         if response.status_code == 422:
#             return _exceptions.UnprocessableEntityError(err_msg, response=response, body=data)

#         if response.status_code == 429:
#             return _exceptions.RateLimitError(err_msg, response=response, body=data)

#         if response.status_code >= 500:
#             return _exceptions.InternalServerError(err_msg, response=response, body=data)
#         return APIStatusError(err_msg, response=response, body=data)


# Client = OpenAI
# AsyncClient = AsyncOpenAI
