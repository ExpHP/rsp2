pub mod version {
    use rsp2_tasks::VersionInfo;

    pub fn get() -> VersionInfo {
        VersionInfo {
            short_sha: option_env!("VERGEN_SHA_SHORT").unwrap_or("(unknown)"),
            commit_date: option_env!("VERGEN_COMMIT_DATE").unwrap_or("(unknown)"),
        }
    }
}
