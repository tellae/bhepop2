"""
This module provides tools to analyse populations.

Most of the time, population analysis is done by comparing
it with reference data.

For enriched populations, comparison with the enrichment source data
can be a good way to assert the quality of the enrichment.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from os import path

from bhepop2.functions import get_feature_from_qualitative_distribution
from bhepop2.sources.base import DEFAULT_SOURCE_NAME


class PopulationAnalysis:
    """
    DISCLAIMER: This class only works with MarginalDistributions data.

    The PopulationAnalysis class and its subclasses were implemented before the
    refactoring of the enrichment classes, which led to the composition
    of SyntheticPopulationEnrichment with EnrichmentSource, which is more generic.
    Therefore, this class expects distributions as in MarginalDistributions.data
    rather than a generic enrichment source data.

    ---------

    Analysis class for synthetic populations.

    Synthetic populations must be identical except for their feature columns.

    The values of the feature columns and their distributions are compared between populations
    and to the reference distribution.

    Analysis is realised on the given modalities, which must be a subset of the modalities used for enrichment
    (and thus available in the population(s) and distributions).

    The following analysis are available:
        - Graphs comparing the distributions in the population(s) to the original distributions (one per modality)
        - A table describing the error of the population(s) in comparison to the distributions (one line per modality), ordered by number of individuals in the modality

    """

    # column describing the analysis class (depends on the feature's type)
    CLASS_COLUMN = None

    # column describing the value corresponding to the class
    VALUE_COLUMN = "value"

    # default format string for the plot title
    DEFAULT_PLOT_TITLE_FORMAT = "Modality {modality} from attribute {attribute}"

    def __init__(
        self,
        populations: dict,
        modalities: dict,
        feature_column: str,
        distributions: pd.DataFrame,
        distributions_name: str = DEFAULT_SOURCE_NAME,
        plot_title_format: str = DEFAULT_PLOT_TITLE_FORMAT,
        output_folder: str = None,
    ):
        """
        Create an analysis instance containing the analysed populations and the reference distributions.

        :param populations: dict containing the analysed populations { pop_name: population_dataframe }
        :param modalities: attribute modalities dict { attribute: [modalities] }
        :param feature_column: name of the column containing the
        :param distributions:
        :param distributions_name:
        :param plot_title_format:
        :param output_folder:
        """
        # analysed data

        # check populations
        if len(populations) == 0:
            raise ValueError("No population to analyse")
        self.populations = populations

        # check distributions DataFrame
        self.distributions = distributions
        self.distributions_name = distributions_name

        # check modalities are consistent with distributions
        self.modalities = modalities

        # check feature column is in populations
        for name, population in populations.items():
            if feature_column not in population.columns:
                raise KeyError(
                    f"Feature column '{feature_column}' not found in '{name}' population"
                )
        self.feature_column = feature_column

        # output fields
        self._output_folder = None
        if output_folder:  # pragma: no cover
            self.set_output_folder(output_folder)

        self.plot_title_format = plot_title_format

        # analysis table
        self._analysis_table = self._evaluate_analysis_table()

    @property
    def analysis_table(self):
        return self._analysis_table

    def set_output_folder(self, output_folder):
        """
        Set a new output folder for this analysis instance.

        :param output_folder: valid output folder path
        """
        if not path.isdir(output_folder):
            raise ValueError(f"{output_folder} is not a folder")
        self._output_folder = output_folder

    def assert_output_folder(self):
        """
        Check that the output folder is set.

        :raises: AssertionError
        """
        assert (
            self._output_folder is not None
        ), "No output folder is set, use set_output_folder method."

    def _evaluate_analysis_table(self):
        """
        Create a table used for comparing populations/distributions.

        The resulting DataFrame contains the following columns:
            - attribute: attribute name
            - modality: modality name
            - self.PROPORTION_COLUMN: value describing the proportion taken for the corresponding
            - one column with the observed_name value
            + one column for each population name

        :return: analysis DataFrame
        """

        # format distributions for analysis
        distributions_df = self._format_distributions_for_analysis()
        distributions_df["source"] = self.distributions_name

        # add population analysis dataframes
        analysis_list = [distributions_df]
        for name, population in self.populations.items():
            population_df = self._compute_distributions_by_attribute(population)
            population_df["source"] = name
            analysis_list.append(population_df)

        # create an analysis dataframe containing all deciles
        analysis_df = pd.concat(analysis_list)

        # pivot data to get final dataframe
        analysis_df = analysis_df.pivot(
            columns="source", index=["attribute", "modality", self.CLASS_COLUMN]
        ).reset_index()
        columns = list(analysis_df.columns.get_level_values(1))
        columns[0], columns[1], columns[2] = "attribute", "modality", self.CLASS_COLUMN
        analysis_df.columns = columns

        return analysis_df

    def _format_distributions_for_analysis(self):
        """
        Format the distributions table for as an analysis table.

        :return: distributions as an analysis table
        """
        raise NotImplementedError

    def _compute_distributions_by_attribute(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the feature values distribution for each modality.

        Generate an analysis table for this population.

        :param population: population DataFrame

        :return: analysis table
        """

        # compute distribution for the whole population
        df_analysis = self._compute_distribution(population)
        df_analysis["attribute"] = "all"
        df_analysis["modality"] = "all"

        # distribution of each modality
        for attribute in self.modalities.keys():
            for modality in self.modalities[attribute]:
                distribution = self._compute_distribution(
                    population[population[attribute] == modality]
                )
                distribution["attribute"] = attribute
                distribution["modality"] = modality

                df_analysis = pd.concat([df_analysis, distribution])
        df_analysis["source"] = "enriched"
        return df_analysis

    def _compute_distribution(self, population: pd.DataFrame):
        """
        Get distribution of the feature values in the population.

        :param population: population DataFrame

        :return: analysis table of the population
        """
        raise NotImplementedError

    def generate_analysis_plots(self):
        """
        Generate plots comparing the population(s) to the original distributions (one per modality).

        Plots are exported to PNG images in the output folder.
        """

        self.assert_output_folder()

        # plot analysis for global distributions
        self.plot_analysis_compare("all", "all").write_image(
            path.join(self._output_folder, "all-all.png")
        )

        # plot analysis by attribute
        for attribute in self.modalities.keys():
            for modality in self.modalities[attribute]:
                self.plot_analysis_compare(attribute, modality).write_image(
                    path.join(self._output_folder, f"{attribute}-{modality}.png")
                )

    def plot_analysis_compare(self, attribute: str, modality: str):
        """
        Generate a plot comparing the populations and the distributions, for the given attribute and modality.

        :param attribute: attribute value
        :param modality: attribute modality

        :return: Plotly Figure
        """
        raise NotImplementedError

    def generate_analysis_error_table(self, export_csv: bool = True):
        """
        Generate a table describing how analysed populations deviate from the original distributions.

        :param export_csv:
        :return:
        """
        raise NotImplementedError

    def get_plot_title(self, **kwargs) -> str:
        """
        Get the plot title for the given keys.

        This on the `plot_title_format` attribute, which can be
        set externally.

        :param kwargs: keys provided to the plot_title_format string

        :return: plot title
        """
        return self.plot_title_format.format(observed_name=self.distributions_name, **kwargs)


class QuantitativeAnalysis(PopulationAnalysis):
    CLASS_COLUMN = "decile"

    def plot_analysis_compare(self, attribute: str, modality: str):
        """
        Comparison plot between reference data and simulation

        :param attribute:
        :param modality:

        :return: Plotly figure
        """

        analysis_table = self._analysis_table
        analysis_table = analysis_table[
            (analysis_table["attribute"] == attribute) & (analysis_table["modality"] == modality)
        ]
        fig = px.line(
            x=analysis_table[self.distributions_name],
            y=analysis_table[self.distributions_name],
            color_discrete_sequence=["black"],
        )

        # add simulated names
        for population_name in self.populations.keys():
            fig.add_trace(
                go.Scatter(
                    x=analysis_table[self.distributions_name],
                    y=analysis_table[population_name],
                    mode="markers",
                    name=population_name,
                )
            )

        # configure plot
        fig.update_layout(
            title=self.get_plot_title(attribute=attribute, modality=modality),
            xaxis_title=f"Observation ({self.distributions_name})",
            yaxis_title="Simulation",
        )

        return fig

    def generate_analysis_error_table(self, export_csv=True):
        """
        Generate a table describing how analysed populations deviate from the original distributions.

        :param export_csv: boolean a csv export should be realised

        :return: error table DataFrame
        """
        self.assert_output_folder()

        analysis_df = self._analysis_table
        # evaluate distance to distributions
        for population_name in self.populations.keys():
            analysis_df[f"{population_name}_perc_error"] = abs(
                (analysis_df[population_name] - analysis_df[self.distributions_name])
                / analysis_df[self.distributions_name]
            )
        error_analysis_df = analysis_df.groupby(["attribute", "modality"], as_index=False).agg(
            {f"{population_name}_perc_error": "mean" for population_name in self.populations.keys()}
        )

        # count number of occurrence of each modality and add it as a "number" column, sort descending
        pop = list(self.populations.values())[0]
        count = [
            pop.groupby([attribute])[attribute].count().rename("number")
            for attribute in self.modalities.keys()
        ]
        count = pd.concat(count)
        count["all"] = len(pop)
        error_analysis_df = error_analysis_df.merge(
            count, how="left", left_on="modality", right_index=True
        )
        error_analysis_df.sort_values(["number"], ascending=False, inplace=True)

        # export to csv if asked
        if export_csv:
            error_analysis_df.to_csv(
                f"{self._output_folder}/analysis_error.csv", sep=";", index=False
            )

        return error_analysis_df

    def _format_distributions_for_analysis(self):
        distributions = self.distributions[
            ["attribute", "modality", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
        ]
        distributions_formated = distributions.melt(
            id_vars=["attribute", "modality"],
            value_vars=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
            value_name=self.VALUE_COLUMN,
            var_name=self.CLASS_COLUMN,
        )
        return distributions_formated

    def _compute_distribution(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        Compute decile distribution of the feature values.

        :param population: analysed population

        :return: dataframe of deciles
        """
        return pd.DataFrame(
            {
                self.VALUE_COLUMN: np.percentile(
                    population[self.feature_column],
                    np.arange(0, 100, 10),
                )[1:],
                self.CLASS_COLUMN: ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
            }
        )


class QualitativeAnalysis(PopulationAnalysis):
    CLASS_COLUMN = "feature"

    def plot_analysis_compare(self, attribute: str, modality: str):
        """
        Comparison plot between reference data and simulation

        :param attribute:
        :param modality:

        :return: Plotly figure
        """

        analysis_table = self._analysis_table
        analysis_table = analysis_table.copy()[
            (analysis_table["attribute"] == attribute) & (analysis_table["modality"] == modality)
        ]

        fig = px.histogram(
            analysis_table,
            x=self.CLASS_COLUMN,
            y=list(self.populations.keys()) + [self.distributions_name],
            barmode="group",
            histnorm="percent",
        )

        # configure plot
        fig.update_layout(
            title=self.get_plot_title(attribute=attribute, modality=modality),
            xaxis_title=self.feature_column,
            yaxis_title="Distribution (%)",
        )

        return fig

    def _format_distributions_for_analysis(self):
        feature_values = get_feature_from_qualitative_distribution(self.distributions)
        distributions = self.distributions[["attribute", "modality"] + feature_values]

        distributions_formated = distributions.melt(
            id_vars=["attribute", "modality"],
            value_vars=feature_values,
            value_name=self.VALUE_COLUMN,
            var_name=self.CLASS_COLUMN,
        )

        return distributions_formated

    def _compute_distribution(self, population: pd.DataFrame) -> pd.DataFrame:
        population.loc[:, "key"] = 1

        res = population.groupby(self.feature_column)["key"].agg("count")

        df = pd.DataFrame({self.CLASS_COLUMN: res.index, self.VALUE_COLUMN: res.values})
        df[self.VALUE_COLUMN] = df[self.VALUE_COLUMN] / df[self.VALUE_COLUMN].sum()

        return df
