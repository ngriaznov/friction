Reviewing a small FastAPI dependency that decodes a bearer JWT and returns the current user, posted with the question "is this a reasonable auth dependency or am I missing something."

The overall shape — a `get_current_user` dependency that pulls the token via `OAuth2PasswordBearer`, decodes with `jwt.decode(token, SECRET_KEY, algorithms=["HS256"])`, and loads the user by the `sub` claim — is a completely standard and reasonable pattern, so nice work on the structure. Two things worth tightening.

You're passing `algorithms=["HS256"]` explicitly, which is good — that's the fix for the classic "alg confusion" issue where a token signed differently than expected gets accepted, so you're already ahead of a common mistake here. Just double check that nothing elsewhere in the codebase decodes tokens without pinning the algorithm list, since it only takes one careless call site to reopen that hole.

The thing I'd actually change: you're catching `jwt.exceptions.DecodeError` but not `jwt.exceptions.ExpiredSignatureError`, and PyJWT raises the latter as a distinct exception rather than folding it into `DecodeError`. As written, an expired token will raise an uncaught exception that FastAPI turns into a 500, when it should be a 401. Catch `jwt.exceptions.PyJWTError` (the common base class for basically all of PyJWT's exceptions, including expired-signature, invalid-signature, and malformed-token cases) instead of the narrower `DecodeError`, and raise your `HTTPException(status_code=401, ...)` from that single except block. That both fixes the bug and simplifies the code.

Small nit: `SECRET_KEY` is read via `os.environ["JWT_SECRET"]` at import time in the dependency module, which means the app will fail fast on missing config — that's actually good — but consider routing it through your existing Pydantic `Settings` object if you have one elsewhere, just for consistency so all config loads through one path.

One more thing to consider, not a bug in what's shown but worth asking: is there a refresh-token flow and a revocation mechanism anywhere? Plain JWT auth like this has no way to invalidate a token before its expiry, so if this is meant to support logout-everywhere or account suspension, you'll need either short expiries plus refresh tokens, or a server-side denylist checked in this same dependency. Not a change request against this specific snippet, just flagging it since it usually comes up right after this code is written.

Overall: approve with the exception-handling fix; everything else is solid.
