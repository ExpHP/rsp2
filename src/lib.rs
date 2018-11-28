pub mod version {
    use rsp2_tasks::VersionInfo;

    pub fn get() -> VersionInfo {
        VersionInfo {
            short_sha: env!("VERGEN_SHA_SHORT"),
            commit_date: env!("VERGEN_COMMIT_DATE"),
        }
    }
}
