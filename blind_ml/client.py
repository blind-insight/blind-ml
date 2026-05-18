"""
Blind Insight Python Client for Jupyter Notebooks

This client allows data scientists to query data from Blind Insight
and use it in machine learning workflows with scikit-learn, pandas, etc.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


@dataclass
class QueryTiming:
    """Timing data for a single query."""

    query_type: str  # 'query' or 'aggregate'
    filter_str: str  # The filter/agg_filter used
    start_time: float
    end_time: float
    network_ms: float  # Time for HTTP request
    parse_ms: float  # Time to parse JSON response
    total_ms: float  # Total time
    success: bool
    error: str | None = None


class ProfilingStats:
    """Collects and reports profiling statistics."""

    def __init__(self):
        self.enabled = False
        self.timings: list[QueryTiming] = []
        self.session_start: float | None = None

    def enable(self):
        """Enable profiling and reset stats."""
        self.enabled = True
        self.timings = []
        self.session_start = time.time()

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def record(self, timing: QueryTiming):
        """Record a query timing."""
        if self.enabled:
            self.timings.append(timing)

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        if not self.timings:
            return {"error": "No timings recorded"}

        query_times = [t for t in self.timings if t.query_type == "query"]
        agg_times = [t for t in self.timings if t.query_type == "aggregate"]

        def stats(timings: list[QueryTiming], field: str) -> dict[str, float]:
            if not timings:
                return {"count": 0, "min": 0, "max": 0, "avg": 0, "total": 0}
            values = [getattr(t, field) for t in timings]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "total": sum(values),
            }

        return {
            "session_duration_sec": time.time() - self.session_start if self.session_start else 0,
            "total_queries": len(self.timings),
            "successful_queries": sum(1 for t in self.timings if t.success),
            "failed_queries": sum(1 for t in self.timings if not t.success),
            "query_timing_ms": stats(query_times, "total_ms"),
            "aggregate_timing_ms": stats(agg_times, "total_ms"),
            "network_ms": stats(self.timings, "network_ms"),
            "parse_ms": stats(self.timings, "parse_ms"),
            "total_ms": stats(self.timings, "total_ms"),
        }

    def breakdown(self) -> dict[str, float]:
        """Get time breakdown by category."""
        if not self.timings:
            return {}

        total_network = sum(t.network_ms for t in self.timings)
        total_parse = sum(t.parse_ms for t in self.timings)
        total_time = sum(t.total_ms for t in self.timings)

        return {
            "network_ms": total_network,
            "network_pct": (total_network / total_time * 100) if total_time > 0 else 0,
            "parse_ms": total_parse,
            "parse_pct": (total_parse / total_time * 100) if total_time > 0 else 0,
            "overhead_ms": total_time - total_network - total_parse,
            "overhead_pct": ((total_time - total_network - total_parse) / total_time * 100) if total_time > 0 else 0,
            "total_ms": total_time,
        }

    def per_query_breakdown(self) -> list[dict[str, Any]]:
        """Get per-query timing details."""
        return [
            {
                "type": t.query_type,
                "filter": t.filter_str[:50] + "..." if len(t.filter_str) > 50 else t.filter_str,
                "network_ms": round(t.network_ms, 2),
                "parse_ms": round(t.parse_ms, 2),
                "total_ms": round(t.total_ms, 2),
                "success": t.success,
            }
            for t in self.timings
        ]


# Global profiling instance
profiling = ProfilingStats()


class BlindInsightClient:
    """
    Client for querying data from Blind Insight via the backend API.

    Example:
        >>> client = BlindInsightClient(api_url="http://localhost:3001")
        >>> data = client.query(
        ...     organization="my-org",
        ...     dataset_slug="iris-dataset",
        ...     schema_slug="iris-schema",
        ...     limit=150
        ... )
        >>> df = client.to_dataframe(data)
        >>> X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        >>> y = df['species'].values
    """

    def __init__(
        self,
        api_url: str = "http://localhost:3001",
        backend: str | None = None,
        proxy_url: str = "http://localhost:3002",
        proxy_auth: tuple | None = None,
        verify_ssl: bool = True,
    ):
        """
        Initialize the Blind Insight client.

        Args:
            api_url: Base URL of the backend API (default: http://localhost:3001)
            backend: "cli" to call the Blind CLI directly, "http" for /api/blind/query,
                "proxy" for direct proxy API calls (fastest), or None to use BI_BACKEND
                env var (defaults to "cli").
            proxy_url: URL of the blind proxy (default: http://localhost:3002)
            proxy_auth: Tuple of (email, password) for proxy auth, or None to use
                BI_EMAIL and BI_PASSWORD env vars. Required for the proxy backend —
                the proxy HTTP API authenticates each request independently of
                ``./blind login``.
            verify_ssl: Whether to verify SSL certificates (default: True).
                Set to False for local dev with self-signed certs.
        """
        self.api_url = api_url.rstrip("/")
        self.proxy_url = proxy_url.rstrip("/")
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.profiling = profiling  # Use global profiling instance
        self.backend = (backend or os.environ.get("BI_BACKEND") or "cli").lower()
        self._blind_path = self._resolve_blind_path()

        # Proxy auth (for "proxy" backend)
        if proxy_auth:
            self._proxy_auth = proxy_auth
        else:
            email = os.environ.get("BI_EMAIL")
            password = os.environ.get("BI_PASSWORD")
            self._proxy_auth = (email, password) if email and password else None

        # Schema ID cache for proxy backend (avoids repeated lookups)
        self._schema_id_cache: dict[str, str] = {}

    def _resolve_blind_path(self) -> Path:
        env_path = os.environ.get("BLIND_PATH")
        if env_path:
            return Path(env_path)
        # Default: Look for blind CLI relative to this file.
        # Set BLIND_PATH environment variable to point to your blind binary,
        # e.g. BLIND_PATH=/usr/local/bin/blind
        return Path(__file__).resolve().parent / "blind"

    def _parse_cli_output(self, output: str) -> Any:
        output = output.strip()
        if not output:
            return []
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            # Prefer the LAST JSON array in the output (CLI logs before JSON)
            for idx in range(len(output) - 1, -1, -1):
                if output[idx] != "[":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(output[idx:])
                    return parsed
                except json.JSONDecodeError:
                    continue
            # Fallback: prefer the last JSON object
            for idx in range(len(output) - 1, -1, -1):
                if output[idx] != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(output[idx:])
                    return parsed
                except json.JSONDecodeError:
                    continue
            # Fallback: scan forward for any JSON (best-effort)
            for idx, ch in enumerate(output):
                if ch not in "[{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(output[idx:])
                    return parsed
                except json.JSONDecodeError:
                    continue
        raise ValueError("Blind CLI did not return JSON. Check CLI output and auth.")

    def _cli_record_list(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        limit: int,
        offset: int,
        filters: list[str] | None,
        decrypt: bool,
    ) -> list[dict[str, Any]]:
        blind_path = self._blind_path
        if not blind_path.exists():
            raise FileNotFoundError(f"Blind CLI not found at {blind_path}. Set BLIND_PATH.")

        args = [
            str(blind_path),
            "record",
            "list",
            f"--organization={organization}",
            f"--dataset={dataset_slug}",
            f"--schema={schema_slug}",
            f"--limit={limit}",
            f"--offset={offset}",
        ]
        filter_str = ",".join(filters) if filters else ""
        if filter_str:
            args.extend(["--filter", filter_str])
        if decrypt:
            args.append("--decrypt")

        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise ValueError(f"Blind CLI failed: {err}")

        parsed = self._parse_cli_output(proc.stdout)
        if isinstance(parsed, dict) and "records" in parsed:
            return parsed.get("records", [])
        if isinstance(parsed, list):
            return parsed
        return []

    def _get_proxy_auth(self):
        """Return HTTPBasicAuth for proxy requests, or raise with a clear message."""
        if not self._proxy_auth:
            raise ValueError(
                "Proxy auth required. Set BI_EMAIL and BI_PASSWORD in your .env file.\n"
                "The proxy HTTP API authenticates each request separately from ./blind login.\n"
                "See .env.example for the template."
            )
        from requests.auth import HTTPBasicAuth

        return HTTPBasicAuth(*self._proxy_auth)

    def _get_schema_id(self, organization: str, dataset_slug: str, schema_slug: str) -> str:
        """Get schema ID from cache or resolve it via API."""
        cache_key = f"{organization}/{dataset_slug}/{schema_slug}"
        if cache_key in self._schema_id_cache:
            return self._schema_id_cache[cache_key]

        auth = self._get_proxy_auth()

        # Get org ID
        resp = self.session.get(f"{self.proxy_url}/api/organizations/by-slug/{organization}/", auth=auth)
        resp.raise_for_status()
        org_id = resp.json()["id"]

        # Get dataset ID
        resp = self.session.get(f"{self.proxy_url}/api/organizations/{org_id}/dataset/{dataset_slug}/", auth=auth)
        resp.raise_for_status()
        dataset_id = resp.json()["id"]

        # Get schema ID
        resp = self.session.get(f"{self.proxy_url}/api/datasets/{dataset_id}/schema/{schema_slug}/", auth=auth)
        resp.raise_for_status()
        schema_id = resp.json()["id"]

        self._schema_id_cache[cache_key] = schema_id
        return schema_id

    def warm_up(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        preflight_aggs: list = None,
        preflight_filters: list = None,
    ) -> None:
        """
        Pre-warm the client by caching schema IDs, establishing connections,
        and running throwaway queries to warm server-side encrypted indexes.

        Args:
            preflight_aggs: Legacy aggregate strings (e.g. "risk_level:count(50~100)")
            preflight_filters: List of filter lists for count_only queries
                               (e.g. [["cancer_5yr:1"], ["cancer_5yr:1", "age_group:50_59"]])
        """
        if self.backend != "proxy":
            return  # Only needed for proxy backend

        if preflight_aggs is None and preflight_filters is None:
            preflight_aggs = [
                "risk_level:count(0~100)",
                "risk_level:count(50~100),fraud_type:wire_transfer",
                "risk_level:count(50~100),account_jurisdiction:us",
                "risk_level:count(50~100),is_active:true",
                "risk_level:count(50~100),month:1",
                "risk_level:count(50~100),reporting_bank_id:BANK001",
                "risk_level:count(50~100),year:2024",
            ]

        import time
        from concurrent.futures import ThreadPoolExecutor

        start = time.time()

        # Pre-cache schema ID
        self._get_schema_id(organization, dataset_slug, schema_slug)
        id_elapsed = (time.time() - start) * 1000

        agg_start = time.time()
        n_queries = 0

        if preflight_filters:

            def _run_filter(filt):
                try:
                    self.query(organization, dataset_slug, schema_slug, filters=filt, limit=1, count_only=True)
                except Exception:
                    pass

            n_queries = len(preflight_filters)
            with ThreadPoolExecutor(max_workers=n_queries) as ex:
                list(ex.map(_run_filter, preflight_filters))

        if preflight_aggs:

            def _run_agg(agg):
                try:
                    self.aggregate(organization, dataset_slug, schema_slug, agg)
                except Exception:
                    pass

            n_queries += len(preflight_aggs)
            with ThreadPoolExecutor(max_workers=len(preflight_aggs)) as ex:
                list(ex.map(_run_agg, preflight_aggs))

        agg_elapsed = (time.time() - agg_start) * 1000
        total = (time.time() - start) * 1000
        print(
            f"  Proxy warm-up: schema ({id_elapsed:.0f}ms) "
            f"+ {n_queries} index preflights ({agg_elapsed:.0f}ms) "
            f"= {total:.0f}ms total"
        )

    def _proxy_query(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        limit: int,
        offset: int,
        filters: list[str] | None,
        decrypt: bool,
        count_only: bool = False,
    ) -> dict[str, Any]:
        """Query records directly via proxy API (fastest backend)."""
        auth = self._get_proxy_auth()
        start_time = time.time()

        # Get schema ID (cached after first call)
        schema_id = self._get_schema_id(organization, dataset_slug, schema_slug)

        # Build search payload
        payload = {
            "schema": schema_id,
            "limit": limit,
            "offset": offset,
        }

        # Convert filter strings to proxy format
        # Handle comma-separated filters (e.g., "field1:value1,field2:value2")
        if filters:
            proxy_filters = []
            for f in filters:
                # Split comma-separated filter strings first
                individual_filters = f.split(",")
                for individual in individual_filters:
                    if ":" in individual:
                        label, value = individual.split(":", 1)
                        proxy_filters.append({"label": label, "value": value})
            if proxy_filters:
                payload["filters"] = proxy_filters

        # Two-phase count_only: POST to get X-Query header, then GET with count_only
        # The proxy encrypts filters on POST but doesn't forward URL params.
        # The server returns X-Query (encrypted q params) in the response header.
        # A follow-up GET through the proxy passthrough delivers count_only to the server.
        if count_only:
            search_url = f"{self.proxy_url}/api/records/search/"
            phase1_payload = {**payload, "limit": 1}
            resp = self.session.post(search_url, json=phase1_payload, auth=auth)
            resp.raise_for_status()
            phase1_end = time.time()

            x_query = resp.headers.get("X-Query", "")
            if not x_query:
                # Fallback: no X-Query means no filters reached the server;
                # use the records response length as a rough count.
                result = resp.json()
                count_val = len(result) if isinstance(result, list) else 0
            else:
                count_url = f"{self.proxy_url}/api/records/?{x_query}&count_only=true"
                count_resp = self.session.get(count_url, auth=auth)
                count_resp.raise_for_status()
                count_data = count_resp.json()
                count_val = int(count_data.get("count", 0))

            end_time = time.time()
            if self.profiling.enabled:
                filter_str = ",".join(filters) if filters else "count_only"
                timing = QueryTiming(
                    query_type="count_only",
                    filter_str=filter_str,
                    start_time=start_time,
                    end_time=end_time,
                    network_ms=(phase1_end - start_time) * 1000,
                    parse_ms=(end_time - phase1_end) * 1000,
                    total_ms=(end_time - start_time) * 1000,
                    success=True,
                    error=None,
                )
                self.profiling.record(timing)
            return {"success": True, "count": count_val, "records": [], "encrypted": True}

        url = f"{self.proxy_url}/api/records/search/"
        resp = self.session.post(url, json=payload, auth=auth)
        resp.raise_for_status()
        result = resp.json()
        network_end = time.time()

        records = result

        # Decrypt if requested
        if decrypt and records:
            decrypt_resp = self.session.post(f"{self.proxy_url}/api/records/decrypt/", json=records, auth=auth)
            decrypt_resp.raise_for_status()
            records = decrypt_resp.json()

        end_time = time.time()

        # Record timing if profiling
        if self.profiling.enabled:
            filter_str = ",".join(filters) if filters else f"limit={limit}"
            timing = QueryTiming(
                query_type="query",
                filter_str=filter_str,
                start_time=start_time,
                end_time=end_time,
                network_ms=(network_end - start_time) * 1000,
                parse_ms=(end_time - network_end) * 1000,
                total_ms=(end_time - start_time) * 1000,
                success=True,
                error=None,
            )
            self.profiling.record(timing)

        # Normalize record format
        normalized = []
        for r in records:
            if isinstance(r, dict):
                if "data" in r:
                    normalized.append(r)
                else:
                    normalized.append({"data": r, "id": r.get("id", "")})
            else:
                normalized.append({"data": r})

        return {
            "success": True,
            "count": len(normalized),
            "records": normalized,
            "encrypted": not decrypt,
        }

    def query(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        limit: int = 1000,
        offset: int = 0,
        filters: list[str] | None = None,
        decrypt: bool = False,
        count_only: bool = False,
    ) -> dict[str, Any]:
        """
        Query records from Blind Insight using encrypted search.

        Blind Insight supports encrypted queries without decryption:
        - Equality: "field:value"
        - Comparisons: "field:>40", "field:<17", "field:>=47", "field:<=17"
        - Ranges: "field:40~45"
        - Aggregations: "field:avg(40~45)", "field:sum(>40)", "field:count(<15)", etc.

        Args:
            organization: Blind Insight organization slug
            dataset_slug: Dataset slug in Blind Insight
            schema_slug: Schema slug in Blind Insight
            limit: Maximum number of records to return (default: 1000)
            offset: Number of records to skip (default: 0)
            filters: List of encrypted filter strings (e.g., ["age:>40", "name:John"])
            decrypt: If True, decrypt data (only needed for ML operations that require plaintext)
            count_only: If True, return only {"count": N} without records (45-200x faster)

        Returns:
            Dictionary containing 'success', 'count', 'records', 'encrypted', etc.

        Raises:
            requests.RequestException: If the API request fails
        """
        if self.backend == "cli":
            start_time = time.time()
            records = self._cli_record_list(
                organization=organization,
                dataset_slug=dataset_slug,
                schema_slug=schema_slug,
                limit=limit,
                offset=offset,
                filters=filters,
                decrypt=decrypt,
            )
            end_time = time.time()

            if self.profiling.enabled:
                filter_str = ",".join(filters) if filters else f"limit={limit}"
                timing = QueryTiming(
                    query_type="query",
                    filter_str=filter_str,
                    start_time=start_time,
                    end_time=end_time,
                    network_ms=(end_time - start_time) * 1000,
                    parse_ms=0.0,
                    total_ms=(end_time - start_time) * 1000,
                    success=True,
                    error=None,
                )
                self.profiling.record(timing)

            return {
                "success": True,
                "count": len(records),
                "records": records,
                "encrypted": not decrypt,
            }

        # Proxy backend - direct API calls (fastest)
        if self.backend == "proxy":
            return self._proxy_query(
                organization=organization,
                dataset_slug=dataset_slug,
                schema_slug=schema_slug,
                limit=limit,
                offset=offset,
                filters=filters,
                decrypt=decrypt,
                count_only=count_only,
            )

        # HTTP backend - calls Blind Insight API
        url = f"{self.api_url}/api/blind/query"

        payload = {
            "organization": organization,
            "datasetSlug": dataset_slug,
            "schemaSlug": schema_slug,
            "limit": limit,
            "offset": offset,
            "filters": filters or [],
            "decrypt": decrypt,
        }

        # Timing instrumentation (only when profiling enabled)
        filter_str = ",".join(filters) if filters else f"limit={limit}"
        start_time = time.time()

        response = self.session.post(url, json=payload)
        response.raise_for_status()
        network_end = time.time()

        result = response.json()
        parse_end = time.time()

        # Record timing if profiling is enabled
        if self.profiling.enabled:
            timing = QueryTiming(
                query_type="query",
                filter_str=filter_str,
                start_time=start_time,
                end_time=parse_end,
                network_ms=(network_end - start_time) * 1000,
                parse_ms=(parse_end - network_end) * 1000,
                total_ms=(parse_end - start_time) * 1000,
                success=True,
                error=None,
            )
            self.profiling.record(timing)

        return result

    def aggregate(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        agg_filter: str,
        extra_filters: list[str] | None = None,
        decrypt: bool = False,
    ) -> dict[str, Any]:
        """
        Run an aggregation query on encrypted data.

        Args:
            organization: Blind Insight organization slug
            dataset_slug: Dataset slug
            schema_slug: Schema slug
            agg_filter: Aggregation expression, e.g. "sepal-length:avg(0~10)" or "petal-width:count(<1.0)"
            extra_filters: Optional list of additional filters (e.g., ["species:I. setosa"])
            decrypt: Should remain False for encrypted aggregation; set True only if you explicitly need plaintext.

        Returns:
            Dictionary containing aggregation result. The aggregation value is typically in records[0]["data"]["value"].
        """
        filters = extra_filters or []
        filters = filters + [agg_filter]

        result = self.query(
            organization=organization,
            dataset_slug=dataset_slug,
            schema_slug=schema_slug,
            limit=1,
            offset=0,
            filters=filters,
            decrypt=decrypt,
        )

        # Update the last timing to be marked as 'aggregate' for profiling
        if self.profiling.enabled and self.profiling.timings:
            self.profiling.timings[-1].query_type = "aggregate"
            self.profiling.timings[-1].filter_str = agg_filter

        return result

    def to_dataframe(self, query_result: dict[str, Any], records_key: str = "records") -> pd.DataFrame:
        """
        Convert query result to a pandas DataFrame.

        Args:
            query_result: Result dictionary from query() method
            records_key: Key in the result dictionary containing records (default: "records")

        Returns:
            pandas DataFrame with the records
        """
        if not query_result.get("success"):
            raise ValueError(f"Query was not successful: {query_result.get('error', 'Unknown error')}")

        records = query_result.get(records_key, [])

        if not records:
            # Return empty DataFrame with proper structure if no records
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(records)

        return df

    def load_data(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        limit: int = 1000,
        offset: int = 0,
        filters: list[str] | None = None,
        decrypt: bool = True,  # Default to True for DataFrame conversion (needs plaintext)
    ) -> pd.DataFrame:
        """
        Convenience method to query and convert to DataFrame in one step.

        Note: This method defaults to decrypt=True because pandas DataFrames require plaintext data.
        For encrypted queries without decryption, use query() directly.

        Args:
            organization: Blind Insight organization slug
            dataset_slug: Dataset slug in Blind Insight
            schema_slug: Schema slug in Blind Insight
            limit: Maximum number of records to return (default: 1000)
            offset: Number of records to skip (default: 0)
            filters: List of encrypted filter strings for encrypted search
            decrypt: If True, decrypt data (default: True for DataFrame conversion)

        Returns:
            pandas DataFrame with the queried records
        """
        result = self.query(organization, dataset_slug, schema_slug, limit, offset, filters, decrypt)
        return self.to_dataframe(result)

    def health_check(self) -> dict[str, Any]:
        """
        Check if the API is available and healthy.

        Returns:
            Dictionary with API status
        """
        if self.backend == "cli":
            raise NotImplementedError("Health check is not available in CLI mode.")
        url = f"{self.api_url}/api/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def load_iris_from_blind(
    organization: str,
    dataset_slug: str,
    schema_slug: str,
    api_url: str = "http://localhost:3001",
    filters: list[str] | None = None,
    decrypt: bool = True,  # Must be True for ML - scikit-learn needs plaintext
) -> tuple:
    """
    Load Iris dataset from Blind Insight in a format compatible with scikit-learn.

    This function mimics sklearn.datasets.load_iris() but loads data from Blind Insight.

    Note: This function requires decrypt=True because scikit-learn needs plaintext data.
    However, you can use encrypted filters to narrow down the dataset before decryption.

    Args:
        organization: Blind Insight organization slug
        dataset_slug: Dataset slug containing the Iris data
        schema_slug: Schema slug containing the Iris data
        api_url: Base URL of the backend API (default: http://localhost:3001)
        filters: Optional list of encrypted filter strings (e.g., ["sepal-length:>5.0"])
        decrypt: Must be True for ML use (scikit-learn requires plaintext) - default: True

    Returns:
        Tuple of (X, y) where:
        - X: numpy array of shape (n_samples, n_features) with feature data
        - y: numpy array of shape (n_samples,) with target labels

    Example:
        >>> # Load all data (decrypted for ML)
        >>> X, y = load_iris_from_blind(
        ...     organization="my-org",
        ...     dataset_slug="iris-dataset",
        ...     schema_slug="iris-schema"
        ... )
        >>>
        >>> # Or use encrypted filters first, then decrypt only filtered results
        >>> X, y = load_iris_from_blind(
        ...     organization="my-org",
        ...     dataset_slug="iris-dataset",
        ...     schema_slug="iris-schema",
        ...     filters=["sepal-length:>5.0"]  # Encrypted filter
        ... )
        >>> # Use X and y with scikit-learn
        >>> from sklearn.linear_model import LogisticRegression
        >>> clf = LogisticRegression()
        >>> clf.fit(X, y)
    """
    client = BlindInsightClient(api_url=api_url)
    df = client.load_data(organization, dataset_slug, schema_slug, limit=150, filters=filters, decrypt=decrypt)

    if df.empty:
        raise ValueError("No data returned from Blind Insight")

    # Expected Iris dataset columns:
    # - sepal_length, sepal_width, petal_length, petal_width (features) - with underscore or hyphen
    # - species or target (label)

    # Try to find feature columns (handle both underscore and hyphen formats, union them)
    feature_cols = []
    underscore_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    hyphen_cols = ["sepal-length", "sepal-width", "petal-length", "petal-width"]

    for col in underscore_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)
    for col in hyphen_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)

    if not feature_cols:
        # If standard names not found, use all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = ["species", "target", "class", "label", "dataset-order", "dataset_order"]
        feature_cols = [col for col in numeric_cols if col.lower() not in [t.lower() for t in target_cols]]

    if not feature_cols:
        raise ValueError("Could not identify feature columns in the dataset")

    # Try to find target column
    target_col = None
    for col in ["species", "target", "class", "label"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError("Could not identify target column (expected: species, target, class, or label)")

    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values

    # Convert target to numeric if it's string
    if y.dtype == object:
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y


def load_fraud_from_blind(
    organization: str,
    dataset_slug: str = "fraud-training",
    schema_slug: str = "fraud-training",
    api_url: str = "http://localhost:3001",
    filters: list[str] | None = None,
    decrypt: bool = True,
    limit: int = 5000,
) -> tuple:
    """
    Load a fraud dataset from Blind Insight for binary classification.

    Data remains stored encrypted; this decrypts only the rows you fetch so that
    scikit-learn can train. Use encrypted filters in `filters` to narrow the
    training set before decryption if needed.

    Args:
        organization: Blind Insight organization slug
        dataset_slug: Dataset slug (default: fraud-training)
        schema_slug: Schema slug (default: fraud-training)
        api_url: Backend API URL
        filters: Optional list of encrypted filter strings
        decrypt: Must remain True for ML (defaults to True)
        limit: Maximum rows to fetch for training

    Returns:
        Tuple of (df, feature_cols, target_col):
        - df: pandas DataFrame with the raw rows
        - feature_cols: list of columns to use as model features
        - target_col: the target column name ("is_fraud")
    """
    client = BlindInsightClient(api_url=api_url)
    df = client.load_data(
        organization,
        dataset_slug,
        schema_slug,
        limit=limit,
        filters=filters,
        decrypt=decrypt,
    )

    if df.empty:
        raise ValueError("No data returned from Blind Insight for fraud dataset")

    target_col = "is_fraud"
    if target_col not in df.columns:
        raise ValueError("Expected target column 'is_fraud' not found in dataset")

    drop_cols = {target_col, "dataset-order", "dataset_order"}
    feature_cols = [col for col in df.columns if col not in drop_cols]

    if not feature_cols:
        raise ValueError("No feature columns detected for fraud dataset")

    return df, feature_cols, target_col


def load_account_risk_from_blind(
    organization: str,
    dataset_slug: str = "account-info",
    schema_slug: str = "account-info",
    api_url: str = "http://localhost:3001",
    filters: list[str] | None = None,
    decrypt: bool = True,
    limit: int = 5000,
    risk_threshold: int = 50,
) -> tuple:
    """
    Load account risk data from Blind Insight for binary classification.

    Derives a binary target from risk_level: high risk (>=threshold) vs low risk (<threshold).
    Data remains stored encrypted; this decrypts only the rows you fetch so that
    scikit-learn can train.

    Args:
        organization: Blind Insight organization slug
        dataset_slug: Dataset slug (default: account-info)
        schema_slug: Schema slug (default: account-info)
        api_url: Backend API URL
        filters: Optional list of encrypted filter strings
        decrypt: Must remain True for ML (defaults to True)
        limit: Maximum rows to fetch for training
        risk_threshold: Threshold for high risk classification (default: 50)

    Returns:
        Tuple of (df, feature_cols, target_col):
        - df: pandas DataFrame with the raw rows (includes derived is_high_risk column)
        - feature_cols: list of columns to use as model features
        - target_col: the target column name ("is_high_risk")
    """
    client = BlindInsightClient(api_url=api_url)
    df = client.load_data(
        organization,
        dataset_slug,
        schema_slug,
        limit=limit,
        filters=filters,
        decrypt=decrypt,
    )

    if df.empty:
        raise ValueError("No data returned from Blind Insight for account risk dataset")

    if "risk_level" not in df.columns:
        raise ValueError("Expected column 'risk_level' not found in dataset")

    # Derive binary target: is_high_risk = 1 if risk_level >= threshold, else 0
    df["is_high_risk"] = (df["risk_level"] >= risk_threshold).astype(int)
    target_col = "is_high_risk"

    # Exclude target, risk_level (used to derive target), and identifier columns
    drop_cols = {target_col, "risk_level", "report_id", "reported_iban", "dataset-order", "dataset_order"}
    feature_cols = [col for col in df.columns if col not in drop_cols]

    if not feature_cols:
        raise ValueError("No feature columns detected for account risk dataset")

    return df, feature_cols, target_col
