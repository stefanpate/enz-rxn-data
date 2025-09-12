import polars as pl

known_compounds_schema = pl.Schema(
    {
        "id": pl.String,
        "smiles": pl.String,
        "name": pl.String,
        "chebi_id": pl.String,
        "n_atoms": pl.Int32,
    }
)

existence_enum = pl.Enum(
    [
        "Predicted",
        "Uncertain",
        "Inferred from homology",
        "Evidence at transcript level",
        "Evidence at protein level",
    ]
)

reviewed_enum = pl.Enum(
    [
        "reviewed",
        "unreviewed",
    ]
)

enzymes_schema = pl.Schema(
    {
        "id": pl.String,
        "sequence": pl.String,
        "existence": existence_enum,
        "reviewed": reviewed_enum,
        "ec": pl.String,
        "organism": pl.String,
        "name": pl.String,
        "subunit": pl.Boolean,
    }
)

known_reactions_schema = pl.Schema(
        {
        "id": pl.String,
        "smarts": pl.String,
        "enzymes": pl.List(pl.String),
        "reverse": pl.String,
        "db_ids": pl.List(pl.String),
    }
)