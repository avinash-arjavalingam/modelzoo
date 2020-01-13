package cmd

import (
	"context"

	"github.com/harbor-ml/modelzoo/go/server"
	"github.com/spf13/cobra"
)

// serveCmd represents the serve command
var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the server",
	Run: func(cmd *cobra.Command, args []string) {
		// Pass a do-nothing ctx for serving
		ctx := context.Background()
		server.ServeForever(ctx, false, 9000, "")
	},
}

var proxyCmd = &cobra.Command{
	Use:   "proxy",
	Short: "Start the reverse HTTP proxy",
	Run: func(cmd *cobra.Command, args []string) {
		server.ProxyForever(9000, 9090)
	},
}

func init() {
	rootCmd.AddCommand(serveCmd)

	rootCmd.AddCommand(proxyCmd)
	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// serveCmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	// serveCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}
